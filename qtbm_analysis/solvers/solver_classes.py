"""
solver_classes.py -- importable solver classes for the benchmark notebook.

Put this file in /solvers, then from the notebook (which lives in /jupyter):

    import sys
    from pathlib import Path
    sys.path.append(str((Path.cwd() / ".." / "solvers").resolve()))
    from solver_classes import (
        SparseLU, GMRES, BlockThomas, BlockThomasExplicitInv, extract_blocks_sparse,  # CPU, always available
        UMFPACK, MUMPS,                                        # CPU, extra installs
        GMRESCuPy, CuDSS,                                      # GPU only
        gpu_available,
    )

Common class contract (mirrors the notebook's original SparseLU):
    __init__(...)       does ALL setup/factorization work (that's what you time)
    solve(b)            b is 1-D or 2-D (n, nrhs); returns same shape, on host
    get_LUP()           explicit factors where the solver exposes them,
                        None where it doesn't (MUMPS, cuDSS -- by design)
    factor_nbytes()     memory footprint of the stored factors (best effort)

Factor availability summary:
    SuperLU              : L, U, perm_r, perm_c            (Pr @ A @ Pc == L @ U)
    UMFPACK              : L, U, perm_r, perm_c, R         (Pr @ diag(1/R) @ A @ Pc == L @ U)
    BlockThomas          : per-block LU factors (Higham Alg. 13.3, Implementation 1)
    BlockThomasExplicitInv: per-block EXPLICIT INVERSES, no LU factors (Higham
                            Alg. 13.3, Implementation 2) -- get_LUP() returns None,
                            use get_inverses() instead.
    MUMPS                : nothing -- factors live in internal Fortran-side structures;
               python-mumps's Context API does not expose them.
    cuDSS    : permutations + factor nnz count only -- factor VALUES are
               opaque GPU-resident structures, not exposed by the public API.
"""

import time
import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------
def gpu_available():
    """True iff CuPy is importable AND at least one CUDA device is visible."""
    try:
        import cupy as cp
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _solve_columns(solve_one, b):
    """Apply a single-RHS solver column-by-column to a 1-D or 2-D b."""
    b = np.asarray(b)
    if b.ndim == 1:
        return solve_one(b)
    return np.column_stack(
        [solve_one(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])]
    )


# ===========================================================================
# 1) SPARSE LU (SuperLU) -- BASELINE  (moved verbatim from the notebook)
# ===========================================================================
class SparseLU:
    def __init__(self, A_csc, dtype=None):
        self.dtype = np.dtype(dtype) if dtype is not None else A_csc.dtype
        self.lu = spla.splu(A_csc.astype(self.dtype).tocsc())

    def solve(self, b):
        return self.lu.solve(b.astype(self.dtype))

    def get_LUP(self):                     # Pr @ A @ Pc == L @ U
        return self.lu.L, self.lu.U, self.lu.perm_r, self.lu.perm_c

    def factor_nbytes(self):
        L, U = self.lu.L, self.lu.U
        return int(L.data.nbytes + L.indices.nbytes + L.indptr.nbytes +
                   U.data.nbytes + U.indices.nbytes + U.indptr.nbytes +
                   self.lu.perm_r.nbytes + self.lu.perm_c.nbytes)


# ===========================================================================
# 2) GMRES (SciPy, CPU, optional ILU preconditioner) -- from the notebook
# ===========================================================================
class GMRES:
    def __init__(self, A_csc, dtype=None, rtol=None, restart=50, maxiter=1000,
                 use_ilu=True):
        self.dtype = np.dtype(dtype) if dtype is not None else A_csc.dtype
        self.A = A_csc.astype(self.dtype).tocsc()
        if rtol is None:                   # pick a reachable tol for the precision
            rtol = 1e-10 if self.dtype.itemsize >= 16 else 1e-6
        self.rtol, self.restart, self.maxiter = rtol, restart, maxiter
        self.M = None
        self._ilu = None
        if use_ilu:
            self._ilu = spla.spilu(self.A)              # "factor" step
            self.M = spla.LinearOperator(self.A.shape, self._ilu.solve)
        self.last_iters = None
        self.last_info = None

    def _solve_one(self, b):
        iters = [0]
        def cb(x): iters[0] += 1
        x, info = spla.gmres(self.A, b.astype(self.dtype), M=self.M,
                             rtol=self.rtol, restart=self.restart,
                             maxiter=self.maxiter, callback=cb,
                             callback_type='legacy')
        self.last_iters, self.last_info = iters[0], info
        return x

    def solve(self, b):
        return _solve_columns(self._solve_one, b)

    def get_LUP(self):
        return None                        # ILU is a preconditioner, not an LU of A

    def factor_nbytes(self):
        if self._ilu is None:
            return 0
        L, U = self._ilu.L, self._ilu.U
        return int(L.data.nbytes + U.data.nbytes + L.indices.nbytes +
                   U.indices.nbytes + L.indptr.nbytes + U.indptr.nbytes)


# ===========================================================================
# 3) BLOCK THOMAS -- from the notebook
#    Higham, "Accuracy and Stability of Numerical Algorithms", Alg. 13.3,
#    IMPLEMENTATION 1: diagonal blocks factorized by GEPP (LAPACK getrf via
#    scipy.linalg.lu_factor); every place the algorithm needs D_mod_k^{-1},
#    it instead does a triangular substitution against the stored LU factors
#    (scipy.linalg.lu_solve). No block is ever explicitly inverted.
# ===========================================================================
class BlockThomas:
    def __init__(self, D, L, U, dtype=None):
        self.N = len(D)
        self.dtype = np.dtype(dtype) if dtype is not None else D[0].dtype
        D = [d.astype(self.dtype) for d in D]
        self.L = [l.astype(self.dtype) for l in L]
        self.U = [u.astype(self.dtype) for u in U]
        self.bs = D[0].shape[0]
        self.D_mod  = [None] * self.N
        self.lu_piv = [None] * self.N
        self.D_mod[0]  = D[0]
        self.lu_piv[0] = sla.lu_factor(D[0])
        for k in range(1, self.N):
            self.D_mod[k]  = D[k] - self.L[k-1] @ sla.lu_solve(self.lu_piv[k-1], self.U[k-1])
            self.lu_piv[k] = sla.lu_factor(self.D_mod[k])

    def solve(self, b):
        was_array = not isinstance(b, list)
        bs = self.bs
        bb = [b[k*bs:(k+1)*bs] for k in range(self.N)] if was_array else [bk.copy() for bk in b]
        bb = [bk.astype(self.dtype) for bk in bb]
        for k in range(1, self.N):
            bb[k] = bb[k] - self.L[k-1] @ sla.lu_solve(self.lu_piv[k-1], bb[k-1])
        x = [None] * self.N
        x[-1] = sla.lu_solve(self.lu_piv[-1], bb[-1])
        for k in range(self.N-2, -1, -1):
            x[k] = sla.lu_solve(self.lu_piv[k], bb[k] - self.U[k] @ x[k+1])
        return np.concatenate(x, axis=0) if was_array else x

    def get_LUP(self):                     # block factors, all same block size
        Lb = np.stack(self.L) if self.L else np.empty((0, self.bs, self.bs), self.dtype)
        Ub = np.stack(self.U) if self.U else np.empty((0, self.bs, self.bs), self.dtype)
        Dlu  = np.stack([lu  for lu, piv in self.lu_piv])
        Dpiv = np.stack([piv for lu, piv in self.lu_piv])
        return Lb, Ub, Dlu, Dpiv

    def factor_nbytes(self):
        Lb, Ub, Dlu, Dpiv = self.get_LUP()
        return int(Lb.nbytes + Ub.nbytes + Dlu.nbytes + Dpiv.nbytes)


# ===========================================================================
# 3b) BLOCK THOMAS, EXPLICIT INVERSE -- Higham Alg. 13.3, IMPLEMENTATION 2
#
#    "A_11 is computed explicitly, so that step 2 becomes a matrix
#    multiplication and Ux = y is solved entirely by matrix-vector
#    multiplications. This approach is attractive for parallel machines."
#
#    Same recursion as BlockThomas (Implementation 1):
#        D_mod[0]     = D[0]
#        D_mod[k]     = D[k] - L[k-1] @ D_mod_inv[k-1] @ U[k-1]
#    but here D_mod_inv[k] = inv(D_mod[k]) is formed EXPLICITLY once per
#    block, and every place Implementation 1 does a lu_solve (substitution),
#    this does a dense matmul with the stored inverse instead. No LU factors
#    or pivots are stored at all -- get_LUP() returns None; use
#    get_inverses() to retrieve the explicit block inverses.
#
#    Numerically this is NOT equivalent to Implementation 1: explicit
#    inversion is more expensive (~3x the flops of an LU factorization for
#    a dense block) and, per Higham Ch. 14, forming A^{-1} explicitly and
#    multiplying is generally less accurate than solving via LU substitution
#    -- though for well-conditioned diagonal blocks the difference is
#    usually modest. It's included here for the flops/parallelism
#    comparison Higham describes, not because it's expected to win on
#    accuracy.
# ===========================================================================
class BlockThomasExplicitInv:
    def __init__(self, D, L, U, dtype=None):
        self.N = len(D)
        self.dtype = np.dtype(dtype) if dtype is not None else D[0].dtype
        D = [d.astype(self.dtype) for d in D]
        self.L = [l.astype(self.dtype) for l in L]
        self.U = [u.astype(self.dtype) for u in U]
        self.bs = D[0].shape[0]
        self.D_mod     = [None] * self.N
        self.D_mod_inv = [None] * self.N
        self.D_mod[0]     = D[0]
        self.D_mod_inv[0] = sla.inv(D[0])
        for k in range(1, self.N):
            self.D_mod[k]     = D[k] - self.L[k-1] @ self.D_mod_inv[k-1] @ self.U[k-1]
            self.D_mod_inv[k] = sla.inv(self.D_mod[k])

    def solve(self, b):
        was_array = not isinstance(b, list)
        bs = self.bs
        bb = [b[k*bs:(k+1)*bs] for k in range(self.N)] if was_array else [bk.copy() for bk in b]
        bb = [bk.astype(self.dtype) for bk in bb]
        # forward sweep: substitution replaced by matmul with D_mod_inv
        for k in range(1, self.N):
            bb[k] = bb[k] - self.L[k-1] @ (self.D_mod_inv[k-1] @ bb[k-1])
        # block back substitution: also entirely matrix-vector multiplications
        x = [None] * self.N
        x[-1] = self.D_mod_inv[-1] @ bb[-1]
        for k in range(self.N-2, -1, -1):
            x[k] = self.D_mod_inv[k] @ (bb[k] - self.U[k] @ x[k+1])
        return np.concatenate(x, axis=0) if was_array else x

    def get_LUP(self):
        return None                        # no LU factors in Implementation 2 -- see docstring

    def get_inverses(self):
        """Explicit block inverses D_mod_inv[k], stacked (N, bs, bs). This is
        the Implementation-2 analogue of get_LUP() -- there are no L/U/pivot
        factors to return, only the explicitly formed inverses."""
        Db_inv = np.stack(self.D_mod_inv) if self.D_mod_inv else \
            np.empty((0, self.bs, self.bs), self.dtype)
        Lb = np.stack(self.L) if self.L else np.empty((0, self.bs, self.bs), self.dtype)
        Ub = np.stack(self.U) if self.U else np.empty((0, self.bs, self.bs), self.dtype)
        return Db_inv, Lb, Ub

    def factor_nbytes(self):
        Db_inv, Lb, Ub = self.get_inverses()
        return int(Db_inv.nbytes + Lb.nbytes + Ub.nbytes)


def extract_blocks_sparse(As, bs):
    """Block-tridiagonal blocks from sparse CSC (densifies only the small blocks)."""
    n = As.shape[0]
    assert n % bs == 0, "n must be divisible by block size"
    N = n // bs
    Ac = As.tocsr()
    D = [np.asarray(Ac[k*bs:(k+1)*bs, k*bs:(k+1)*bs].todense())     for k in range(N)]
    L = [np.asarray(Ac[(k+1)*bs:(k+2)*bs, k*bs:(k+1)*bs].todense()) for k in range(N-1)]
    U = [np.asarray(Ac[k*bs:(k+1)*bs, (k+1)*bs:(k+2)*bs].todense()) for k in range(N-1)]
    return D, L, U


# ===========================================================================
# 4) UMFPACK  (CPU, double precision ONLY)
# ===========================================================================
class UMFPACK:
    """
    Sparse LU via scikit-umfpack.  conda install -c conda-forge scikit-umfpack

    UMFPACK only supports double precision (float64 / complex128) -- there is
    no complex64 build.  Passing a single-precision dtype raises TypeError so
    the benchmark stays honest instead of silently upcasting.

    Explicit factors ARE available.  Confirmed reconstruction convention
    (empirically verified to machine precision, see solvers/umfpack_ex.py):

        Pr @ diag(1/R) @ A @ Pc == L @ U

    with Pr built from argsort(perm_r), Pc built directly from perm_c,
    and R the row-scaling vector (rows of A divided by R before permuting).
    """

    _DOUBLE = (np.dtype(np.float64), np.dtype(np.complex128))

    def __init__(self, A_csc, dtype=None):
        import scikits.umfpack as um
        self.dtype = np.dtype(dtype) if dtype is not None else A_csc.dtype
        if self.dtype not in self._DOUBLE:
            raise TypeError(
                f"UMFPACK only supports double precision (float64/complex128), "
                f"got {self.dtype.name}. Skip UMFPACK for single-precision runs."
            )
        self.lu = um.splu(A_csc.astype(self.dtype).tocsc())

    def _solve_one(self, b):
        return self.lu.solve(b.astype(self.dtype))

    def solve(self, b):
        # scikit-umfpack's splu.solve is single-RHS -> loop columns for 2-D b
        return _solve_columns(self._solve_one, b)

    def get_LUP(self):                     # Pr @ diag(1/R) @ A @ Pc == L @ U
        return self.lu.L, self.lu.U, self.lu.perm_r, self.lu.perm_c

    def get_scaling(self):
        """Row-scaling vector R and the do_recip flag (None if absent)."""
        return getattr(self.lu, "R", None), getattr(self.lu, "do_recip", None)

    def factor_nbytes(self):
        L, U = self.lu.L, self.lu.U
        nbytes = int(L.data.nbytes + L.indices.nbytes + L.indptr.nbytes +
                     U.data.nbytes + U.indices.nbytes + U.indptr.nbytes +
                     np.asarray(self.lu.perm_r).nbytes +
                     np.asarray(self.lu.perm_c).nbytes)
        R, _ = self.get_scaling()
        if R is not None:
            nbytes += np.asarray(R).nbytes
        return nbytes


# ===========================================================================
# 5) MUMPS  (CPU)
# ===========================================================================
class MUMPS:
    """
    Sparse LU via python-mumps.  conda install -c conda-forge python-mumps

    dtype of the matrix drives MUMPS's internal precision choice, so both
    complex128 and complex64 work.

    *** NO EXPLICIT FACTORS ***
    MUMPS keeps L/U in internal (Fortran-side, possibly MPI-distributed)
    structures; python-mumps's high-level Context API exposes no .L/.U/.perm_*.
    get_LUP() therefore returns None.  MUMPS does have a native JOB=7/JOB=8
    save/restore for factorizations, but python-mumps doesn't wrap it -- in
    practice, "persisting" a MUMPS factorization means re-running factor().
    """

    def __init__(self, A, dtype=None, symmetric=False):
        import mumps  # python-mumps
        self.dtype = np.dtype(dtype) if dtype is not None else A.dtype
        A = A.astype(self.dtype).tocsc()
        self.ctx = mumps.Context()
        self.ctx.set_matrix(A, symmetric=symmetric)
        self.ctx.factor()

    def solve(self, b):
        b = np.asarray(b, dtype=self.dtype)
        try:
            return self.ctx.solve(b)               # some versions accept 2-D b
        except Exception:
            return _solve_columns(self.ctx.solve, b)

    def get_LUP(self):
        return None                                # not exposed -- see docstring

    def factor_nbytes(self):
        # Best effort: MUMPS reports factor size via INFOG(3) (total number of
        # factor entries; negative means value is in millions).  Whether this
        # array is reachable depends on the python-mumps version, so fall back
        # to 0 rather than crash.
        try:
            infog = self.ctx.mumps_instance.infog
            n_entries = int(infog[2])              # INFOG(3), 0-based index 2
            if n_entries < 0:
                n_entries = -n_entries * 1_000_000
            return int(n_entries * self.dtype.itemsize)
        except Exception:
            return 0


# ===========================================================================
# 6) GMRES (CuPy, GPU only)
# ===========================================================================
class GMRESCuPy:
    """
    GMRES on the GPU via cupyx.scipy.sparse.linalg.gmres.
    pip install cupy-cuda12x   (or cupy-cuda13x)

    Raises RuntimeError at construction if no CUDA device is visible, so a
    CPU-only machine skips this solver cleanly.

    "Factorization" here is the host->device transfer of A (there is no
    preconditioner -- cupyx has no spilu), so factor time == transfer time.
    """

    def __init__(self, A, dtype=None, rtol=None, restart=50, maxiter=1000):
        if not gpu_available():
            raise RuntimeError("GMRESCuPy requires CuPy and a visible CUDA GPU.")
        import cupy as cp
        import cupyx.scipy.sparse as cusp
        self._cp = cp
        self.dtype = np.dtype(dtype) if dtype is not None else A.dtype
        if rtol is None:
            rtol = 1e-10 if self.dtype.itemsize >= 16 else 1e-6
        self.rtol, self.restart, self.maxiter = rtol, restart, maxiter

        self.A_gpu = cusp.csr_matrix(A.astype(self.dtype).tocsr())
        cp.cuda.Stream.null.synchronize()          # transfer is async -- wait
        self.last_iters = None
        self.last_info = None

    def _solve_one_gpu(self, b_gpu):
        from cupyx.scipy.sparse.linalg import gmres
        iters = [0]
        def cb(x): iters[0] += 1
        x_gpu, info = gmres(self.A_gpu, b_gpu, rtol=self.rtol,
                            atol=0.0, restart=self.restart,
                            maxiter=self.maxiter, callback=cb,
                            callback_type="pr_norm")
        self.last_iters, self.last_info = iters[0], int(info)
        return x_gpu

    def solve(self, b):
        cp = self._cp
        b_gpu = cp.asarray(np.asarray(b, dtype=self.dtype))
        if b_gpu.ndim == 1:
            x_gpu = self._solve_one_gpu(b_gpu)
        else:
            cols, total_iters = [], 0
            for j in range(b_gpu.shape[1]):
                cols.append(self._solve_one_gpu(cp.ascontiguousarray(b_gpu[:, j])))
                total_iters += self.last_iters
            x_gpu = cp.column_stack(cols)
            self.last_iters = total_iters
        cp.cuda.Stream.null.synchronize()          # GPU is async -- wait before timing stops
        return cp.asnumpy(x_gpu)

    def get_LUP(self):
        return None                                # iterative -- no factors

    def factor_nbytes(self):
        return 0                                   # no factors stored (A itself isn't a factor)


# ===========================================================================
# 7) cuDSS  (GPU only, via nvmath-python's DirectSolver)
# ===========================================================================
class CuDSS:
    """
    Sparse direct solve on the GPU via nvmath.sparse.advanced.DirectSolver.
    pip install nvmath-python[cu12]   (or [cu13])

    Raises RuntimeError at construction if no CUDA device is visible.

    *** NO EXPLICIT FACTOR VALUES (same category as MUMPS) ***
    cuDSS keeps the numerical L/U in opaque GPU-resident structures.  What IS
    exposed (and what get_metadata() returns) is:
        col_permutation / row_permutation  (from plan/reordering)
        lu_nnz                             (nnz count of the factors)
    If you need reconstructable explicit L/U, use SuperLU or UMFPACK.

    *** PARTIALLY UNVERIFIED (no GPU available while writing this) ***
    - DirectSolver binds b at construction; changing the RHS afterwards is
      attempted via solver.reset_operands(b=...).  If your nvmath version
      doesn't support that, solve() falls back to rebuilding + refactorizing
      with the new b (correct result, but factor/solve timing separation is
      lost -- a warning is printed).
    - 'row_permutation' attribute name is not independently confirmed;
      get_metadata() uses getattr and simply omits missing entries.
    Run once and report any AttributeError/TypeError so names can be fixed.
    """

    def __init__(self, A, dtype=None, nrhs=1):
        if not gpu_available():
            raise RuntimeError("CuDSS requires an NVIDIA GPU with a working CUDA install.")
        import nvmath
        self.dtype = np.dtype(dtype) if dtype is not None else A.dtype
        # DirectSolver wants CSR; the shared matrices are CSC -- convert here.
        self.A_csr = A.astype(self.dtype).tocsr()
        self.n = self.A_csr.shape[0]
        self.nrhs = int(nrhs)

        b0 = np.zeros((self.n, self.nrhs), dtype=self.dtype, order="F")
        self.solver = nvmath.sparse.advanced.DirectSolver(self.A_csr, b0)
        self.plan_info = self.solver.plan()        # reordering + symbolic
        self.fac_info = self.solver.factorize()    # numerical factorization

    def solve(self, b):
        b = np.asarray(b, dtype=self.dtype, order="F")
        try:
            self.solver.reset_operands(b=b)        # UNVERIFIED kwarg name
            return np.asarray(self.solver.solve())
        except (AttributeError, TypeError) as e:
            print(f"CuDSS: reset_operands unavailable ({e}); rebuilding solver "
                  f"with the new RHS -- factor/solve timing split is lost for "
                  f"this call.")
            import nvmath
            self.free()
            self.solver = nvmath.sparse.advanced.DirectSolver(self.A_csr, b)
            self.plan_info = self.solver.plan()
            self.fac_info = self.solver.factorize()
            return np.asarray(self.solver.solve())

    def get_LUP(self):
        return None                                # factor values not exposed -- see docstring

    def get_metadata(self):
        md = {
            "col_permutation": getattr(self.plan_info, "col_permutation", None),
            "row_permutation": getattr(self.plan_info, "row_permutation", None),
            "lu_nnz":          getattr(self.fac_info, "lu_nnz", None),
        }
        return {k: v for k, v in md.items() if v is not None}

    def factor_nbytes(self):
        # Values-only estimate from the reported factor nnz (index arrays are
        # internal/unknown), 0 if lu_nnz isn't exposed on this version.
        lu_nnz = getattr(self.fac_info, "lu_nnz", None)
        return int(lu_nnz * self.dtype.itemsize) if lu_nnz is not None else 0

    def free(self):
        try:
            self.solver.free()
        except Exception:
            pass