"""
Sparse direct and iterative solvers behind one common interface.

Purpose
-------
This module is the solver library of the project. Every solver benchmarked,
analysed or refined elsewhere is defined here, so that a comparison between two
solvers is a comparison between two entries of this file and nothing else.

Interface
---------
Every class satisfies the same contract:

    solver = SolverClass(A_or_blocks, dtype)
        Performs ALL setup and factorization work. This is the quantity
        reported as factorization time.
    x = solver.solve(b)
        b is (n,) or (n, nrhs); the return has the same shape and resides in
        host memory, whatever device the solver used internally.
    solver.get_LUP()
        Explicit factors where the solver exposes them, None where it does not.
    solver.factor_nbytes()
        Memory footprint of the stored factors, best effort.

Factor availability
-------------------
    SparseLU (SuperLU)   L, U, perm_r, perm_c
                         Pr A Pc == L U
    UMFPACK              L, U, perm_r, perm_c, R
                         Pr diag(1/R) A Pc == L U
    BlockThomas          per-block LU factors, A == L U exactly
    BlockThomasFP16      the same, embedded-real fp16 at block size 2*bs
    BlockThomasExplicitInv,
    BlockThomasExplicitInvFP16
                         per-block explicit inverses; no LU factors exist, so
                         get_LUP() returns None and get_inverses() is used
    MUMPS                none. The factors live in Fortran-side structures that
                         python-mumps's Context API does not expose.
    CuDSS                permutations and the factor nonzero count only. The
                         factor values are opaque device-resident structures.
    GMRES, GMRESCuPy     none; iterative. The SciPy variant holds an ILU
                         preconditioner, which is not an LU of A.

Block partitions
----------------
All four Block Thomas variants infer their partition from the shapes of the
diagonal blocks they are handed, so no variant takes a block-size argument and
a partition enters the pipeline at exactly one place:

    D, L, U = extract_blocks_sparse(A, block_sizes)

block_sizes is an int for a uniform partition or a sequence of per-block sizes
summing to n for a custom one. block_sizes_from_matrix() derives such a
sequence from the sparsity pattern; offband_nnz(A, sizes) == 0 must be checked
before it is used, because a partition that cuts a real coupling does not fail
loudly.

For a uniform partition get_LUP() and get_inverses() return stacked
(N, bs, bs) arrays. For a custom partition the blocks are ragged and they
return lists of per-block arrays instead.

Contents
--------
    gpu_available()                              CUDA availability probe
    find_block_slices, block_sizes_from_matrix   partition detection
    normalize_block_sizes, block_offsets         partition bookkeeping
    offband_nnz                                  partition validity check
    extract_blocks_sparse                        block extraction
    SparseLU, UMFPACK, MUMPS, GMRES              CPU solvers
    GMRESCuPy, CuDSS                             GPU solvers
    BlockThomas, BlockThomasExplicitInv          Block Thomas, complex
    BlockThomasFP16, BlockThomasExplicitInvFP16  Block Thomas, half precision

Reference
---------
N. J. Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed., SIAM
2002. Algorithm 13.3 and its two implementations; Chapter 14 on the accuracy
of explicit inversion.
"""

import time

import numpy as np
import scipy.linalg as sla
import scipy.sparse.linalg as spla


# ---------------------------------------------------------------------------
# utilities
# ---------------------------------------------------------------------------
def gpu_available():
    """True if CuPy is importable and at least one CUDA device is visible."""
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def _nbytes(part):
    """
    Byte count of a factor part, which is a stacked array for a uniform
    partition and a list of per-block arrays for a custom one.
    """
    if isinstance(part, list):
        return int(sum(np.asarray(a).nbytes for a in part))
    return int(np.asarray(part).nbytes)


def _solve_columns(solve_one, b):
    """Apply a single-RHS solve function column by column to a 1-D or 2-D b."""
    b = np.asarray(b)
    if b.ndim == 1:
        return solve_one(b)
    return np.column_stack(
        [solve_one(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])]
    )


def _mask_from_bits(bits: int = 52) -> np.uint64:
    assert 0 < bits <= 52, f"bits must be in (0, 52], got {bits}"
    total_bits = bits + 11 + 1  # mantissa + exponent + sign
    return np.uint64(((1 << total_bits) - 1) << (64 - total_bits))


def _mask_real_precision(x, mask):
    assert x.dtype == np.float64, f"x must be float64, got {x.dtype}"
    if mask == 52:  # fp64 mantissa bits
        pass
    else:
        mask = _mask_from_bits(mask)
        x = (x.view(np.uint64) & mask).view(np.float64)
    return x


def _mask_complex_precision(x, mask_real, mask_imag):
    x_real = _mask_real_precision(np.real(x), mask_real)
    x_imag = _mask_real_precision(np.imag(x), mask_imag)
    return x_real + 1j * x_imag


def _mask_precision(x, mask):
    if x.dtype != np.complex128:
        return x
    if np.iscomplexobj(x):
        return _mask_complex_precision(x, mask, mask)
    return _mask_real_precision(x, mask)


# ===========================================================================
# 1) Sparse LU via SuperLU. The baseline every other solver is measured
#    against: partial pivoting with a column ordering, in double or single
#    complex precision.
# ===========================================================================
class SparseLU:
    """
    Sparse LU through scipy.sparse.linalg.splu, which wraps SuperLU.

    Pivoting is partial with a fill-reducing column ordering, so the factors
    satisfy Pr A Pc == L U. Both complex128 and complex64 are supported.
    """

    def __init__(self, A_csc, dtype=None):
        self.dtype = np.dtype(dtype) if dtype is not None else A_csc.dtype
        self.lu = spla.splu(A_csc.astype(self.dtype).tocsc())

    def solve(self, b):
        return self.lu.solve(b.astype(self.dtype))

    def get_LUP(self):
        """(L, U, perm_r, perm_c) with Pr A Pc == L U, Pr from argsort(perm_r)."""
        return self.lu.L, self.lu.U, self.lu.perm_r, self.lu.perm_c

    def factor_nbytes(self):
        L, U = self.lu.L, self.lu.U
        return int(
            L.data.nbytes
            + L.indices.nbytes
            + L.indptr.nbytes
            + U.data.nbytes
            + U.indices.nbytes
            + U.indptr.nbytes
            + self.lu.perm_r.nbytes
            + self.lu.perm_c.nbytes
        )


# ===========================================================================
# 2) GMRES on the CPU, through SciPy, with an optional ILU preconditioner.
# ===========================================================================
class GMRES:
    """
    Restarted GMRES with an optional incomplete-LU preconditioner.

    The construction step, which is what the benchmark reports as
    factorization time, computes the ILU factors. The ILU is a preconditioner
    and not an LU factorization of A, so get_LUP() returns None.

    The default tolerance is chosen per precision: 1e-10 in double, 1e-6 in
    single. Requesting a tolerance below the unit roundoff of the working
    precision cannot be met and would only exhaust maxiter.
    """

    def __init__(
        self, A_csc, dtype=None, rtol=None, restart=50, maxiter=1000, use_ilu=True
    ):
        self.dtype = np.dtype(dtype) if dtype is not None else A_csc.dtype
        self.A = A_csc.astype(self.dtype).tocsc()
        if rtol is None:
            rtol = 1e-10 if self.dtype.itemsize >= 16 else 1e-6
        self.rtol, self.restart, self.maxiter = rtol, restart, maxiter
        self.M = None
        self._ilu = None
        if use_ilu:
            self._ilu = spla.spilu(self.A)
            self.M = spla.LinearOperator(self.A.shape, self._ilu.solve)
        self.last_iters = None
        self.last_info = None

    def _solve_one(self, b):
        iters = [0]

        def cb(x):
            iters[0] += 1

        x, info = spla.gmres(
            self.A,
            b.astype(self.dtype),
            M=self.M,
            rtol=self.rtol,
            restart=self.restart,
            maxiter=self.maxiter,
            callback=cb,
            callback_type="legacy",
        )
        self.last_iters, self.last_info = iters[0], info
        return x

    def solve(self, b):
        return _solve_columns(self._solve_one, b)

    def get_LUP(self):
        """None: the ILU is a preconditioner, not a factorization of A."""
        return None

    def factor_nbytes(self):
        if self._ilu is None:
            return 0
        L, U = self._ilu.L, self._ilu.U
        return int(
            L.data.nbytes
            + U.data.nbytes
            + L.indices.nbytes
            + U.indices.nbytes
            + L.indptr.nbytes
            + U.indptr.nbytes
        )


# ===========================================================================
# BLOCK STRUCTURE: uniform and custom partitions
#
# Every Block Thomas variant infers its partition from the shapes of the
# diagonal blocks it receives, so a non-uniform partition enters the pipeline
# at one place only, extract_blocks_sparse(A, block_sizes). block_sizes is an
# int for a uniform partition or a sequence of per-block sizes for a custom
# one.
#
# The detection algorithm lives here rather than in the analysis script that
# exposes it, so that the detector and the solvers consuming its output cannot
# diverge.
# ===========================================================================
def find_block_slices(a):
    """
    Detect the block-tridiagonal structure of a sparse matrix.

    Input
    -----
    a : sparse matrix. Converted to CSR and its indices sorted internally, so
        callers need not do either.

    Algorithm
    ---------
    A reach frontier is grown row by row. Starting from the first row, the
    frontier is the largest column index reachable from any row not yet
    assigned to a block. A block boundary is declared as soon as the frontier
    stops advancing, that is, as soon as no row in the current unresolved range
    references a column beyond what has already been reached. This is the
    standard connected-interval decomposition of a banded pattern.

    The first slice produced is always the single seed row; it is merged into
    the second, so the leading block is coarser than the true structure. A
    coarser partition remains valid, at the cost of slightly more arithmetic in
    the first block.

    Output
    ------
    list of slice objects tiling range(n).

    Limitation
    ----------
    The frontier looks forward only, using the largest column index of each
    row. The returned partition is therefore guaranteed block tridiagonal only
    when the matrix is structurally symmetric, which holds for the QTBM
    matrices but is not enforced. Confirm with offband_nnz(A, sizes) == 0
    before passing the partition to a Block Thomas solver.
    """
    a = a.tocsr()
    a.sort_indices()
    n = a.shape[0]
    block_slices = []
    visited_max = -1
    frontier_max = 0

    while frontier_max > visited_max:
        start = visited_max + 1
        stop = frontier_max + 1
        block_slices.append(slice(start, stop))

        next_max = frontier_max
        for node in range(start, stop):
            row_start = a.indptr[node]
            row_stop = a.indptr[node + 1]
            if row_stop > row_start:
                row_max = int(a.indices[row_stop - 1])
                next_max = max(next_max, row_max)

        visited_max = frontier_max
        frontier_max = min(next_max, n - 1)

    # The first slice is always the single seed row; merge it into the second.
    if len(block_slices) > 1:
        block_slices[1] = slice(block_slices[0].start, block_slices[1].stop)
        block_slices.pop(0)

    return block_slices


def block_sizes_from_matrix(A):
    """find_block_slices() expressed as a tuple of block sizes."""
    return tuple(s.stop - s.start for s in find_block_slices(A))


def normalize_block_sizes(n, block_sizes):
    """
    Validate a partition specification and return it as a tuple of sizes.

    block_sizes is an int, meaning a uniform partition that must divide n, or
    a sequence of positive per-block sizes that must sum to n.

    Raises ValueError on any partition that does not tile the matrix.
    Truncating silently would return a wrong solution rather than an error.
    """
    if np.isscalar(block_sizes):
        bs = int(block_sizes)
        if n % bs:
            raise ValueError(
                f"n={n} is not divisible by block size {bs}; "
                f"pass an explicit list of block sizes instead"
            )
        return (bs,) * (n // bs)
    sizes = tuple(int(s) for s in block_sizes)
    if any(s <= 0 for s in sizes):
        raise ValueError(f"all block sizes must be positive, got {sizes}")
    if sum(sizes) != n:
        raise ValueError(f"block sizes sum to {sum(sizes)}, expected n={n}")
    return sizes


def block_offsets(block_sizes):
    """Cumulative block boundaries, length N + 1; block k is [off[k], off[k+1])."""
    return np.concatenate([[0], np.cumsum(np.asarray(block_sizes, dtype=np.int64))])


def offband_nnz(A, block_sizes):
    """
    Count stored entries outside the block-tridiagonal band of a partition.

    A return value of zero means A is block tridiagonal under this partition,
    which is the precondition every Block Thomas variant assumes. Any other
    value means extract_blocks_sparse would discard real couplings and the
    solve would return a plausible but wrong solution without raising.
    """
    sizes = normalize_block_sizes(A.shape[0], block_sizes)
    off = block_offsets(sizes)
    C = A.tocoo()
    bi = np.searchsorted(off, C.row, side="right") - 1
    bj = np.searchsorted(off, C.col, side="right") - 1
    return int(np.count_nonzero(np.abs(bi - bj) > 1))


def extract_blocks_sparse(As, block_sizes):
    """
    Extract the block-tridiagonal blocks of a sparse matrix.

    Only the individual blocks are densified, never the whole matrix.
    block_sizes is an int for a uniform partition or a sequence summing to n.

    Returns (D, L, U) where D has N diagonal blocks and L, U have N - 1
    off-diagonal blocks each. For a non-uniform partition the off-diagonal
    blocks are rectangular: L[k] is (n[k+1], n[k]) and U[k] is (n[k], n[k+1]).

    Entries outside the band are discarded silently; validate the partition
    with offband_nnz first.
    """
    n = As.shape[0]
    sizes = normalize_block_sizes(n, block_sizes)
    off = block_offsets(sizes)
    N = len(sizes)
    Ac = As.tocsr()
    D = [
        np.asarray(Ac[off[k] : off[k + 1], off[k] : off[k + 1]].todense())
        for k in range(N)
    ]
    L = [
        np.asarray(Ac[off[k + 1] : off[k + 2], off[k] : off[k + 1]].todense())
        for k in range(N - 1)
    ]
    U = [
        np.asarray(Ac[off[k] : off[k + 1], off[k + 1] : off[k + 2]].todense())
        for k in range(N - 1)
    ]
    return D, L, U


class _BlockThomasBase:
    """
    Block bookkeeping shared by all four Block Thomas variants.

    Provides partition inference, shape validation, and the splitting and
    rejoining of a flat right-hand side. The partition is inferred from the
    diagonal blocks, so no variant takes a block-size argument: the blocks
    returned by extract_blocks_sparse fully determine it.
    """

    def _init_blocks(self, D, L, U):
        """
        Record the partition implied by D, L, U and validate every shape.

        Sets N, block_sizes, offsets, n, uniform and bs. Raises ValueError on
        a non-square diagonal block, a wrong number of off-diagonal blocks, or
        an off-diagonal block whose shape is inconsistent with its neighbours.
        """
        if len(L) != len(D) - 1 or len(U) != len(D) - 1:
            raise ValueError(
                f"expected {len(D) - 1} off-diagonal blocks, "
                f"got len(L)={len(L)}, len(U)={len(U)}"
            )
        self.N = len(D)
        self.block_sizes = tuple(int(d.shape[0]) for d in D)
        for k, d in enumerate(D):
            if d.shape[0] != d.shape[1]:
                raise ValueError(f"diagonal block {k} is not square: {d.shape}")
        for k in range(self.N - 1):
            want_L = (self.block_sizes[k + 1], self.block_sizes[k])
            want_U = (self.block_sizes[k], self.block_sizes[k + 1])
            if L[k].shape != want_L:
                raise ValueError(f"L[{k}] has shape {L[k].shape}, expected {want_L}")
            if U[k].shape != want_U:
                raise ValueError(f"U[{k}] has shape {U[k].shape}, expected {want_U}")
        self.offsets = block_offsets(self.block_sizes)
        self.n = int(self.offsets[-1])
        self.uniform = len(set(self.block_sizes)) == 1
        # bs remains an int for a uniform partition, which is what factor_io
        # writes as the block_size attribute; None signals a ragged partition.
        self.bs = self.block_sizes[0] if self.uniform else None

    def _split(self, b):
        """Split a flat (n,) or (n, nrhs) right-hand side into per-block views."""
        off = self.offsets
        return [b[off[k] : off[k + 1]] for k in range(self.N)]

    def _stack(self, blocks, dtype):
        """
        Stack per-block arrays into one (N, ...) array for a uniform partition,
        or return the list unchanged for a ragged one. A uniform run therefore
        keeps the stacked on-disk factor layout that factor_io and the analysis
        scripts read directly.
        """
        if not blocks:
            return np.empty((0, 0, 0), dtype)
        return np.stack(blocks) if self.uniform else list(blocks)


# ===========================================================================
# 3) BLOCK THOMAS, IMPLEMENTATION 1
#    Higham, Accuracy and Stability of Numerical Algorithms, Algorithm 13.3.
# ===========================================================================
class BlockThomas(_BlockThomasBase):
    """
    Block Thomas with LU factorization of the diagonal blocks.

    Input
    -----
    D, L, U : block-tridiagonal blocks of A, as returned by
              extract_blocks_sparse. D has N square diagonal blocks, L the
              N - 1 subdiagonal blocks, U the N - 1 superdiagonal blocks.
    dtype   : working precision; defaults to that of D[0].

    Algorithm
    ---------
    The block LU factorization of a block-tridiagonal matrix

        A = [ D_0  U_0                    ]
            [ L_0  D_1  U_1               ]
            [      L_1  D_2  ...          ]
            [                ...  U_{N-2} ]
            [           L_{N-2}  D_{N-1}  ]

    is A = L_global U_global with

        L_global = block-bidiagonal(I;      E_k = L_{k-1} D_mod_{k-1}^-1)
        U_global = block-bidiagonal(D_mod;  U_k)

    where the modified diagonal blocks follow the Schur-complement recursion

        D_mod_0 = D_0
        D_mod_k = D_k - L_{k-1} D_mod_{k-1}^-1 U_{k-1},    k = 1..N-1.

    In this implementation each D_mod_k is factorized by Gaussian elimination
    with partial pivoting (LAPACK getrf through scipy.linalg.lu_factor), and
    every occurrence of D_mod_k^-1 is realized as a triangular substitution
    against the stored factors (scipy.linalg.lu_solve). No block is ever
    explicitly inverted.

    The solve is the corresponding two-sweep substitution:

        forward     y_0 = b_0,  y_k = b_k - L_{k-1} D_mod_{k-1}^-1 y_{k-1}
        backward    x_{N-1} = D_mod_{N-1}^-1 y_{N-1}
                    x_k     = D_mod_k^-1 (y_k - U_k x_{k+1})

    Cost and stability
    ------------------
    Factorization is N block LU factorizations plus 2(N-1) block products, so
    O(N m^3) for uniform block size m, against O(n^3) for a dense LU of the
    whole matrix. Pivoting is confined to each diagonal block, which is weaker
    than the global partial pivoting of SuperLU or UMFPACK; whether that costs
    stability on these matrices is measured by block-thomas/growth_factor.py.

    Output
    ------
    solve(b) returns an array of the same shape as b, in the working precision.
    get_LUP() returns (L, U, Dmod_lu, Dmod_piv).
    """

    def __init__(self, D, L, U, dtype=None):
        self._init_blocks(D, L, U)

        self.dtype = (
            np.dtype(dtype)
            if dtype is not None and type(dtype) is not int
            else D[0].dtype
        )
        self.D = [d.astype(self.dtype) for d in D]
        self.L = [lb.astype(self.dtype) for lb in L]
        self.U = [u.astype(self.dtype) for u in U]

        if dtype is not None and type(dtype) is int:
            self.bits = dtype
            print(
                f"BlockThomas: using {self.bits}-bit mantissa for factorization and solve"
            )
            self.D = [_mask_precision(d, self.bits) for d in D]
            self.L = [_mask_precision(lb, self.bits) for lb in self.L]
            self.U = [_mask_precision(u, self.bits) for u in self.U]
        else:
            self.bits = 52

        self.D_mod = [None] * self.N
        self.lu_piv = [None] * self.N
        self.D_mod[0] = self.D[0]
        self.lu_piv[0] = sla.lu_factor(self.D[0])
        self.lu_piv[0] = (
            _mask_precision(self.lu_piv[0][0], self.bits),
            self.lu_piv[0][1],
        )
        for k in range(1, self.N):
            tmp = _mask_precision(
                sla.lu_solve(self.lu_piv[k - 1], self.U[k - 1]), self.bits
            )
            tmp = _mask_precision(self.L[k - 1] @ tmp, self.bits)
            self.D_mod[k] = _mask_precision(self.D[k] - tmp, self.bits)
            # self.D_mod[k] = self.D[k] - self.L[k-1] @ sla.lu_solve(self.lu_piv[k-1], self.U[k-1])
            self.lu_piv[k] = sla.lu_factor(self.D_mod[k])
            self.lu_piv[k] = (
                _mask_precision(self.lu_piv[k][0], self.bits),
                self.lu_piv[k][1],
            )

    def solve(self, b):
        """
        Two-sweep block substitution. b is a flat array or a list of per-block
        arrays; the return has the same form.
        """
        was_array = not isinstance(b, list)
        bb = self._split(b) if was_array else [bk.copy() for bk in b]
        bb = [bk.astype(self.dtype) for bk in bb]
        bb = [_mask_precision(bk, self.bits) for bk in bb]
        for k in range(1, self.N):  # forward sweep
            tmp = _mask_precision(
                sla.lu_solve(self.lu_piv[k - 1], bb[k - 1]), self.bits
            )
            tmp = _mask_precision(self.L[k - 1] @ tmp, self.bits)
            bb[k] = _mask_precision(bb[k] - tmp, self.bits)
            # bb[k] = bb[k] - self.L[k-1] @ sla.lu_solve(self.lu_piv[k-1], bb[k-1])
        x = [None] * self.N
        x[-1] = sla.lu_solve(self.lu_piv[-1], bb[-1])
        x[-1] = _mask_precision(x[-1], self.bits)
        for k in range(self.N - 2, -1, -1):  # backward sweep
            tmp = _mask_precision(self.U[k] @ x[k + 1], self.bits)
            tmp = _mask_precision(bb[k] - tmp, self.bits)
            x[k] = _mask_precision(sla.lu_solve(self.lu_piv[k], tmp), self.bits)
            # x[k] = sla.lu_solve(self.lu_piv[k], bb[k] - self.U[k] @ x[k+1])
        return np.concatenate(x, axis=0) if was_array else x

    def get_LUP(self):
        """
        (L, U, Dmod_lu, Dmod_piv), stacked (N, bs, bs) arrays for a uniform
        partition and lists of per-block arrays for a custom one.
        """
        Lb = self._stack(self.L, self.dtype)
        Ub = self._stack(self.U, self.dtype)
        Dlu = self._stack([lu for lu, piv in self.lu_piv], self.dtype)
        Dpiv = self._stack([piv for lu, piv in self.lu_piv], np.int32)
        return Lb, Ub, Dlu, Dpiv

    def factor_nbytes(self):
        return int(sum(_nbytes(part) for part in self.get_LUP()))


# ===========================================================================
# 3b) BLOCK THOMAS, IMPLEMENTATION 2
#     Higham, Algorithm 13.3, second implementation.
# ===========================================================================
class BlockThomasExplicitInv(_BlockThomasBase):
    """
    Block Thomas with explicit inversion of the modified diagonal blocks.

    Input
    -----
    Identical to BlockThomas: D, L, U from extract_blocks_sparse, and a
    working precision.

    Algorithm
    ---------
    The factorization recursion is the same as in implementation 1,

        D_mod_0 = D_0
        D_mod_k = D_k - L_{k-1} D_mod_{k-1}^-1 U_{k-1},

    but D_mod_k^-1 is formed explicitly once per block and stored. Every
    operation that implementation 1 performs as a triangular substitution
    becomes a dense matrix product against that stored inverse. No LU factors
    or pivot vectors are retained, so get_LUP() returns None and the factors
    are retrieved through get_inverses().

    Higham's motivation is stated directly: with the inverse available, the
    forward sweep becomes a matrix multiplication and the back substitution is
    carried out entirely by matrix-vector products. That makes the solve a
    sequence of GEMM calls, which is the form throughput-oriented hardware
    executes most efficiently, and it is why this variant is the one carried
    into half precision.

    Cost and accuracy
    -----------------
    Explicit inversion of a dense m x m block costs roughly three times the
    flops of its LU factorization, so factorization is more expensive than
    implementation 1 while the solve is cheaper per right-hand side. By Higham
    Chapter 14, forming an inverse and multiplying is in general less accurate
    than solving by substitution, because the computed inverse satisfies no
    small residual bound of the form used for triangular solves. For
    well-conditioned diagonal blocks the difference is modest.

    Output
    ------
    solve(b) returns an array of the same shape as b.
    get_LUP() returns None.
    get_inverses() returns (D_mod_inv, D_mod, L, U).
    """

    def __init__(self, D, L, U, dtype=None):
        self._init_blocks(D, L, U)
        self.dtype = np.dtype(dtype) if dtype is not None else D[0].dtype
        D = [d.astype(self.dtype) for d in D]
        self.L = [lb.astype(self.dtype) for lb in L]
        self.U = [u.astype(self.dtype) for u in U]
        self.D_mod = [None] * self.N
        self.D_mod_inv = [None] * self.N
        self.D_mod[0] = D[0]
        self.D_mod_inv[0] = sla.inv(D[0])
        for k in range(1, self.N):
            self.D_mod[k] = D[k] - self.L[k - 1] @ self.D_mod_inv[k - 1] @ self.U[k - 1]
            self.D_mod_inv[k] = sla.inv(self.D_mod[k])

    def solve(self, b):
        """
        Two-sweep block solve in which every stage is a dense matrix product.
        b is a flat array or a list of per-block arrays; the return has the
        same form.
        """
        was_array = not isinstance(b, list)
        bb = self._split(b) if was_array else [bk.copy() for bk in b]
        bb = [bk.astype(self.dtype) for bk in bb]
        # Forward sweep: substitution replaced by multiplication by D_mod_inv.
        for k in range(1, self.N):
            bb[k] = bb[k] - self.L[k - 1] @ (self.D_mod_inv[k - 1] @ bb[k - 1])
        # Backward sweep: also entirely matrix-vector products.
        x = [None] * self.N
        x[-1] = self.D_mod_inv[-1] @ bb[-1]
        for k in range(self.N - 2, -1, -1):
            x[k] = self.D_mod_inv[k] @ (bb[k] - self.U[k] @ x[k + 1])
        return np.concatenate(x, axis=0) if was_array else x

    def get_LUP(self):
        """None: implementation 2 forms no LU factors. See get_inverses()."""
        return None

    def get_inverses(self):
        """
        (D_mod_inv, D_mod, L, U), the implementation 2 analogue of get_LUP().

        Stacked (N, bs, bs) arrays for a uniform partition, lists of per-block
        arrays for a custom one.

        D_mod is returned in addition to its inverse because the growth-factor
        analysis needs it: implementation 2 never forms LU factors, but the
        same global L and U exist mathematically and D_mod is the diagonal of
        the global U.
        """
        Db_inv = self._stack(self.D_mod_inv, self.dtype)
        Db = self._stack(self.D_mod, self.dtype)
        Lb = self._stack(self.L, self.dtype)
        Ub = self._stack(self.U, self.dtype)
        return Db_inv, Db, Lb, Ub

    def factor_nbytes(self):
        # D_mod is retained for the analysis scripts and is not part of what
        # implementation 2 needs in order to solve. It is excluded here so the
        # reported figure remains comparable with implementation 1.
        Db_inv, _Db, Lb, Ub = self.get_inverses()
        return int(_nbytes(Db_inv) + _nbytes(Lb) + _nbytes(Ub))


# ===========================================================================
# 3c) HALF-PRECISION BLOCK THOMAS, both implementations
#
# COMPLEX ARITHMETIC IN A REAL FORMAT
# IEEE binary16 has no complex counterpart, and NumPy provides no complex32.
# Complex blocks are therefore handled through the exact real embedding
#
#       z = a + bi  ->  [[a, -b], [b, a]],
#
# which is a ring isomorphism: it maps complex addition and multiplication
# onto their real matrix counterparts exactly, introducing no error of its
# own. A complex m x m block becomes a real 2m x 2m block and a rectangular
# (p, q) coupling block becomes (2p, 2q), so a custom non-uniform partition is
# carried over unchanged. The cost is a factor of two in each dimension, hence
# four in storage and eight in the flops of a block factorization, relative to
# a native complex half-precision format if one existed.
#
# WHERE THE ARITHMETIC IS AND IS NOT HALF PRECISION
# Every quantity stored by these classes is float16: the embedded off-diagonal
# blocks, the packed LU factors or the explicit inverses, and the intermediate
# right-hand sides during a solve. Every arithmetic result is rounded to
# float16 before being used again. Three categories of operation are
# nevertheless not performed in float16, and they must be distinguished
# because only the last of them affects accuracy.
#
#   1. Accumulation inside a product. NumPy evaluates a float16 operation by
#      promoting to float32, computing, and rounding the result back to
#      float16. A matrix product therefore accumulates in float32 and rounds
#      once per output entry, rather than rounding after every fused multiply
#      and add. This is not an approximation of half precision that has been
#      chosen for convenience: it is exactly the accumulation model of tensor
#      cores and of the mixed-precision GEMM instructions such a
#      factorization would target in practice, so the simulated arithmetic
#      matches the hardware being modelled. It is, however, more accurate than
#      a hypothetical implementation that rounded to float16 after every
#      scalar addition.
#
#   2. Scalings. Every scale factor applied by this module, the global s, the
#      per-block inverse scales t, the matmul guard scale q, and the
#      right-hand-side scale, is an exact power of two chosen by _pow2_scale.
#      Multiplying by a power of two only changes the exponent field, so in
#      the absence of overflow or underflow it is exact in binary floating
#      point. These multiplications are carried out in float32 or float64,
#      because the reciprocal of a scale is frequently subnormal or infinite
#      in float16 and would be destroyed by the format before it could be
#      applied. Since the operations are exact, computing them in a wider
#      format changes no digit of any result; it only prevents the format from
#      losing the factor. The same applies to the row-sum norms used to choose
#      the guard scale, computed in float32: they select an exponent and enter
#      no result.
#
#   3. Formation of the explicit inverse in implementation 2. This is the one
#      genuine departure. BlockThomasExplicitInvFP16 takes an inv_dtype
#      parameter that defaults to float32: the modified diagonal block is
#      promoted, inverted by LAPACK, and the result rounded back to float16
#      for storage and application. Explicit inversion is the least stable
#      step in the algorithm, and performing it in float16 loses accuracy that
#      no subsequent refinement recovers, while the higher-precision inversion
#      is O(N) small dense inversions and changes neither the amount of data
#      stored nor the precision of the solve. Passing inv_dtype=np.float16
#      selects a factorization that is half precision throughout, so that the
#      value of the higher-precision inversion is measured rather than
#      assumed. Implementation 1 has no such parameter: it is half precision
#      in all three categories above, subject only to point 1.
#
# In summary, BlockThomasFP16 is a half-precision factorization with
# float32 accumulation inside products; BlockThomasExplicitInvFP16 is the same
# except that, at its default setting, the inverses are formed in float32.
#
# RANGE
# The normal range of binary16 is [6.10e-5, 65504], roughly five decades,
# against 616 decades for binary64. QTBM blocks do not lie inside it, and the
# intermediate quantities of the algorithm lie there even less often, so three
# separate power-of-two scalings are applied. They are described at their
# points of use: the global scale s in _BlockThomasFP16Base._init_fp16, the
# matmul overflow guard in _matmul16_scaled, and the per-block inverse scales
# t in BlockThomasExplicitInvFP16.
#
# BACKEND
# The float16 kernels are written against NumPy, which makes them portable and
# auditable but orders of magnitude slower than LAPACK. They exist to measure
# accuracy, not time. Substituting a torch or CuPy backend is mechanical.
# ===========================================================================

H = np.float16

# Target magnitude for the largest entry after the global scaling. 1024 leaves
# a factor of 64 below the binary16 overflow threshold of 65504.
_FP16_TARGET = 1024.0

# Target magnitude for a guarded matrix product, leaving a factor of 4.
_FP16_PROD_TARGET = 16384.0


def embed_block(Z):
    """
    Real embedding of a complex block: (p, q) complex -> (2p, 2q) real.

    Uses z = a + bi -> [[a, -b], [b, a]], which preserves products and sums
    exactly, so the embedding introduces no numerical error.
    """
    a, b = np.ascontiguousarray(Z.real), np.ascontiguousarray(Z.imag)
    return np.block([[a, -b], [b, a]])


def embed_vec(z):
    """Real embedding of a complex vector: (m,) complex -> (2m,) real."""
    return np.concatenate([z.real, z.imag], axis=0)


def unembed_vec(v):
    """Inverse of embed_vec: (2m,) real -> (m,) complex."""
    m = v.shape[0] // 2
    return v[:m] + 1j * v[m:]


def _pow2_scale(amax, target=_FP16_TARGET):
    """
    Largest power of two t satisfying amax * t <= target.

    Restricting scale factors to powers of two makes the scaling exact in
    binary floating point: only the exponent field changes, so no digit of the
    significand is lost and the scaling contributes no rounding error.

    Raises FloatingPointError when amax is zero or not finite, since no finite
    scale brings such a block into range.
    """
    if not np.isfinite(amax) or amax == 0.0:
        raise FloatingPointError(f"cannot scale a block with max |entry| = {amax}")
    return 2.0 ** np.floor(np.log2(target / amax))


def _inf_norm(A):
    """
    Maximum absolute row sum, the constant in the bound

        |(A X)_ij| <= ||A||_inf * max|X|,

    which is what the overflow guard in _matmul16_scaled uses to choose a
    scale. Computed in float32; the value selects an exponent and does not
    enter any result.
    """
    return float(np.abs(A.astype(np.float32)).sum(axis=1).max())


def _matmul16_scaled(Ah, a_inf, X):
    """
    Half-precision matrix product guarded against intermediate overflow.

    Input
    -----
    Ah    : (m, k) float16 array.
    a_inf : ||Ah||_inf, precomputed once when Ah was stored.
    X     : (k,) or (k, r) float16 array.

    Motivation
    ----------
    A dense half-precision product of two arrays whose entries sit near the top
    of the scaled range overflows purely because of the accumulation length,
    independently of the magnitude of the result the algorithm needs. With
    entries near 1e3 and an accumulation of length 800 the exact intermediate
    is near 1e9, far past the binary16 ceiling of 65504, even when the wanted
    quantity is O(1). This is reached at the block sizes the QTBM materials
    use: bs = 416 embeds to an accumulation of length 832.

    Algorithm
    ---------
    X is rescaled by a power of two q chosen from the rigorous bound

        |(Ah X)_ij| <= ||Ah||_inf * max|X|,

    so that the product provably cannot leave the representable range. The
    rescaling is exact, so it changes no digit of the result; it only moves
    where the computation sits in the exponent range.

    Output
    ------
    (P, q) with P approximately q * (Ah X) in float16 and q an exact power of
    two. The scale is returned rather than undone in place so that a caller
    owing a further power-of-two factor, as implementation 2 does with its
    per-block inverse scale t, can fold both into a single multiplication.
    Undoing them in sequence would materialize an intermediate that overflows
    on its own.
    """
    xmax = float(np.abs(X).max())
    if xmax == 0.0 or a_inf == 0.0:
        return np.zeros((Ah.shape[0],) + X.shape[1:], dtype=H), 1.0
    q = _pow2_scale(a_inf * xmax, target=_FP16_PROD_TARGET)
    Xq = (X.astype(np.float32) * np.float32(q)).astype(H)
    return (Ah @ Xq).astype(H), q


def _unscale16(P, q):
    """
    Divide out a power-of-two scale and round back to float16.

    Performed in float32 because 1/q is frequently subnormal, or infinite, in
    float16 and would be destroyed before it could be applied. The operation
    is exact, so the wider format changes no digit of the result.
    """
    return (P.astype(np.float32) * np.float32(1.0 / q)).astype(H)


def _matmul16(Ah, a_inf, X):
    """Guarded float16 product Ah X with the guard scale already removed."""
    return _unscale16(*_matmul16_scaled(Ah, a_inf, X))


def lu_fp16(Dm):
    """
    LU factorization with partial pivoting of a float16 matrix.

    Right-looking Gaussian elimination with row pivoting, written out
    explicitly so that every intermediate is rounded to float16 before being
    reused: the multipliers, the rank-one update of the trailing submatrix, and
    the stored factors. Products accumulate in float32 per NumPy semantics, as
    described in the section header.

    Input
    -----
    Dm : (k, k) array, cast to float16 internally; the argument is not modified.

    Output
    ------
    (LU, piv) in the packed LAPACK convention: the strict lower triangle holds
    the multipliers with an implicit unit diagonal, the upper triangle holds U,
    and piv[j] is the row interchanged with row j at step j.

    Raises
    ------
    ZeroDivisionError if a pivot underflows to exactly zero in float16, which
    is a real possibility at this precision even for a nonsingular block.
    """
    LU = Dm.astype(H, copy=True)
    k = LU.shape[0]
    piv = np.arange(k)
    for j in range(k):
        p = j + int(np.argmax(np.abs(LU[j:, j])))
        if p != j:
            LU[[j, p]] = LU[[p, j]].copy()
            piv[[j, p]] = piv[[p, j]].copy()
        if LU[j, j] == 0:
            raise ZeroDivisionError(f"zero fp16 pivot at column {j}")
        LU[j + 1 :, j] = (LU[j + 1 :, j] / LU[j, j]).astype(H)
        LU[j + 1 :, j + 1 :] = (
            LU[j + 1 :, j + 1 :] - np.outer(LU[j + 1 :, j], LU[j, j + 1 :])
        ).astype(H)
    return LU, piv


def lu_solve_fp16(LU, piv, Rhs):
    """
    Solve L U X = P Rhs in float16 from the packed factors of lu_fp16.

    Forward substitution against the unit lower triangle followed by backward
    substitution against U, with every intermediate rounded to float16. Rhs is
    (k,) or (k, r) and the return has the corresponding shape.
    """
    X = Rhs.astype(H)[piv].copy()
    one_d = X.ndim == 1
    if one_d:
        X = X[:, None]
    k = LU.shape[0]
    for j in range(k):  # unit lower triangle, forward
        X[j + 1 :] = (X[j + 1 :] - np.outer(LU[j + 1 :, j], X[j])).astype(H)
    for j in range(k - 1, -1, -1):  # upper triangle, backward
        X[j] = (X[j] / LU[j, j]).astype(H)
        X[:j] = (X[:j] - np.outer(LU[:j, j], X[j])).astype(H)
    return X[:, 0] if one_d else X


def inv_fp16(Dm):
    """
    Explicit inverse of a float16 block, computed entirely in float16.

    Factorizes with lu_fp16 and solves against the identity. Used only when
    BlockThomasExplicitInvFP16 is asked for a factorization that is half
    precision throughout, that is, inv_dtype=np.float16.
    """
    return lu_solve_fp16(*lu_fp16(Dm), np.eye(Dm.shape[0], dtype=H))


class _BlockThomasFP16Base(_BlockThomasBase):
    """
    Setup shared by both half-precision variants: real embedding, global
    scaling, right-hand-side handling, and the return to complex double
    precision.
    """

    def _init_fp16(self, D, L, U):
        """
        Embed and scale the off-diagonal blocks, and return the conversion to
        be applied to the diagonal blocks.

        Global scaling
        --------------
        A single power-of-two scale s is chosen so that the largest entry of
        the embedded matrix lands near _FP16_TARGET = 1024, leaving a factor of
        64 below the binary16 overflow threshold. The maximum is taken over the
        diagonal and both off-diagonals: the coupling blocks may carry the
        largest entries of a QTBM matrix, and scaling on the diagonal alone
        would allow them to overflow.

        Because s is an exact power of two, the scaled matrix s A differs from
        A only in its exponents. All factorization quantities are therefore
        computed for s A, and the final solution is divided by s in _finish.

        Returns the closure to16 mapping a complex block to its scaled,
        embedded float16 form, so the subclasses apply exactly the same
        conversion to their diagonal blocks.
        """
        self._init_blocks(D, L, U)
        self.dtype = H
        amax = max(
            max(float(np.abs(b).max()) for b in D),
            max((float(np.abs(b).max()) for b in L), default=0.0),
            max((float(np.abs(b).max()) for b in U), default=0.0),
        )
        self.s = _pow2_scale(amax)

        def to16(X):
            """Embed and scale a complex block to float16."""
            return (embed_block(X) * self.s).astype(H)

        # to16 = lambda X: (embed_block(X) * self.s).astype(H)
        self.L = [to16(lb) for lb in L]
        self.U = [to16(u) for u in U]
        # Row-sum norms for the overflow guard in _matmul16, computed once per
        # stored block rather than at every application.
        self.L_inf = [_inf_norm(lb) for lb in self.L]
        self.U_inf = [_inf_norm(u) for u in self.U]
        return to16

    def _split_embed(self, b, rs):
        """Right-hand side scaled by rs, split per block, embedded, in float16."""
        return [embed_vec(bk * rs).astype(H) for bk in self._split(b)]

    def solve(self, b):
        """
        Solve for one or several right-hand sides.

        Columns are solved independently, because each requires its own scale
        factor: a single scale shared across columns of very different
        magnitude would push the smaller ones into the subnormal range.
        """
        b = np.asarray(b)
        if b.ndim == 1:
            return self._solve_one(b)
        return np.column_stack(
            [self._solve_one(np.ascontiguousarray(b[:, j])) for j in range(b.shape[1])]
        )

    def _rhs_scale(self, b):
        """
        Power-of-two scale bringing the right-hand side to order unity.

        Binary16 underflows below 6.1e-5 and QTBM right-hand sides are
        routinely far smaller, so an unscaled right-hand side would lose most
        of its significant digits on conversion. Returns None for a zero
        right-hand side, for which no scale exists and the solution is zero.
        """
        rmax = float(np.abs(b).max())
        return None if rmax == 0 else _pow2_scale(rmax, target=1.0)

    def _finish(self, x_blocks, rs):
        """
        Reassemble the solution: unembed, promote to complex double precision,
        and remove both scalings.

        The factors approximate (s A)^-1 = A^-1 / s, and the right-hand side
        carried a factor rs, so the computed blocks must be multiplied by
        s / rs. Both are exact powers of two.
        """
        out = np.concatenate([unembed_vec(xk.astype(np.float64)) for xk in x_blocks])
        return out * (self.s / rs)


class BlockThomasFP16(_BlockThomasFP16Base):
    """
    Implementation 1 of Block Thomas in half precision.

    Input
    -----
    D, L, U : lists of complex blocks, the same inputs BlockThomas takes, from
              extract_blocks_sparse under a uniform or custom partition.
    dtype   : accepted and ignored; the working precision is always float16.

    Algorithm
    ---------
    The recursion and the two-sweep solve are those of BlockThomas, executed on
    the scaled real embedding s A at block size 2m instead of on A at block
    size m. Diagonal blocks are factorized by lu_fp16 and every occurrence of
    D_mod^-1 is a substitution through lu_solve_fp16; no block is inverted.
    Products against the stored off-diagonal blocks pass through the overflow
    guard _matmul16.

    Precision
    ---------
    Half precision throughout. Every stored quantity and every arithmetic
    result is float16; the only wider arithmetic is the float32 accumulation
    inside products, which is the tensor-core model, and the exact
    power-of-two scalings. See the section header.

    Output
    ------
    solve(b) returns complex128, since the scalings are removed at full
    precision and the result is no longer restricted to the half-precision
    range.
    get_LUP() returns embedded-real float16 factors at block size 2m.
    """

    def __init__(self, D, L, U, dtype=None):
        to16 = self._init_fp16(D, L, U)
        self.lu_piv = [None] * self.N
        self.D_mod = [None] * self.N
        Dm = to16(D[0])
        self.D_mod[0] = Dm
        self.lu_piv[0] = lu_fp16(Dm)
        for k in range(1, self.N):
            # D_mod[k] = D[k] - L[k-1] D_mod[k-1]^-1 U[k-1], the inverse
            # applied as a substitution against the stored factors.
            W = lu_solve_fp16(*self.lu_piv[k - 1], self.U[k - 1])
            Dm = (to16(D[k]) - _matmul16(self.L[k - 1], self.L_inf[k - 1], W)).astype(H)
            self.D_mod[k] = Dm
            self.lu_piv[k] = lu_fp16(Dm)

    def _solve_one(self, b):
        """Two-sweep block substitution for a single right-hand side."""
        rs = self._rhs_scale(b)
        if rs is None:
            return np.zeros_like(b)
        bb = self._split_embed(b, rs)
        for k in range(1, self.N):  # forward sweep
            w = lu_solve_fp16(*self.lu_piv[k - 1], bb[k - 1])
            bb[k] = (bb[k] - _matmul16(self.L[k - 1], self.L_inf[k - 1], w)).astype(H)
        x = [None] * self.N
        x[-1] = lu_solve_fp16(*self.lu_piv[-1], bb[-1])
        for k in range(self.N - 2, -1, -1):  # backward sweep
            x[k] = lu_solve_fp16(
                *self.lu_piv[k],
                (bb[k] - _matmul16(self.U[k], self.U_inf[k], x[k + 1])).astype(H),
            )
        return self._finish(x, rs)

    def get_LUP(self):
        """
        (L, U, Dmod_lu, Dmod_piv), embedded-real float16 at block size 2m.

        These are factors of s * embed(A), not of A. Reconstruction requires
        self.s and the recorded partition; see block-thomas/growth_factor.py.
        """
        Lb = self._stack(self.L, H)
        Ub = self._stack(self.U, H)
        Dlu = self._stack([lu for lu, piv in self.lu_piv], H)
        Dpiv = self._stack([piv for lu, piv in self.lu_piv], np.int32)
        return Lb, Ub, Dlu, Dpiv

    def factor_nbytes(self):
        return int(sum(_nbytes(part) for part in self.get_LUP()))


class BlockThomasExplicitInvFP16(_BlockThomasFP16Base):
    """
    Implementation 2 of Block Thomas in half precision.

    Input
    -----
    D, L, U   : lists of complex blocks, as for BlockThomasFP16.
    dtype     : accepted and ignored; storage and application are float16.
    inv_dtype : precision in which the explicit inverses are formed, one of
                float16, float32 (default) or float64.

    Algorithm
    ---------
    The recursion is that of BlockThomasExplicitInv, executed on the scaled
    real embedding s A. Each modified diagonal block is inverted explicitly and
    the inverse stored, so that both the forward sweep and the back
    substitution consist solely of dense matrix products. That is the reason
    this variant is carried into half precision: a solve built from GEMM calls
    is the form that half-precision hardware executes efficiently, whereas the
    triangular substitutions of implementation 1 are not.

    Per-block rescaling of the inverses
    -----------------------------------
    Implementation 1 requires only the global scale s, because every quantity
    it stores has the magnitude of A. Implementation 2 stores inverses, whose
    entries scale like 1 / |D_mod|, that is, in the opposite direction. The
    same global scale that brings A into the half-precision range therefore
    drives its inverses towards the underflow threshold, and a single global
    scale cannot serve both.

    Each inverse consequently carries its own power-of-two scale t[k], stored
    beside it:

        G[k] = t[k] * D_mod_hat[k]^-1,      in float16,

    where hatted quantities are the scaled embedded ones, D_hat = s D and so
    on. Since t[k] is an exact power of two, the rescaling contributes no
    rounding error. The recursion remains consistent in the hatted quantities,

        D_mod_hat[k] = D_hat[k]
                       - (1 / t[k-1]) L_hat[k-1] G[k-1] U_hat[k-1].

    Precision, and where it is not half
    -----------------------------------
    Storage and application are float16 throughout. The single deliberate
    exception is the formation of the inverse:

        inv_dtype = np.float32, the default
            The modified diagonal block is promoted to float32, inverted by
            LAPACK, and the result rounded back to float16 for storage and
            application. This is the standard mixed-precision split, factor
            high and store and apply low. Explicit inversion is the least
            stable step in the algorithm, and performing it in float16 loses
            accuracy that no subsequent step recovers, whereas the float32
            inversion consists of O(N) small dense inversions and leaves both
            the stored data and the precision of the solve unchanged. Under
            this setting the factorization is NOT purely half precision.

        inv_dtype = np.float16
            The inverse is formed by lu_fp16 followed by substitution against
            the identity, so the factorization is half precision throughout.
            Slower and markedly less accurate. It exists so that the value of
            the higher-precision inversion is measured rather than assumed;
            run_bench/sweep_fp16.py --inv-dtype float16 performs that
            measurement.

    Output
    ------
    solve(b) returns complex128.
    get_LUP() returns None; the factors are obtained through get_inverses().
    """

    def __init__(self, D, L, U, dtype=None, inv_dtype=np.float32):
        to16 = self._init_fp16(D, L, U)
        self.inv_dtype = np.dtype(inv_dtype)
        if self.inv_dtype not in (
            np.dtype(np.float16),
            np.dtype(np.float32),
            np.dtype(np.float64),
        ):
            raise ValueError(f"inv_dtype must be float16/32/64, got {self.inv_dtype}")

        self.G = [None] * self.N  # scaled explicit inverses, float16
        self.t = np.ones(self.N)  # their power-of-two scales
        self.G_inf = [0.0] * self.N  # row-sum norms for the matmul guard
        self.D_mod = [None] * self.N  # retained for the growth-factor analysis

        Dm = to16(D[0])
        self.D_mod[0] = Dm
        self._store_inverse(0, Dm)
        for k in range(1, self.N):
            V = self._apply_ginv(k - 1, self.U[k - 1])
            Dm = (to16(D[k]) - _matmul16(self.L[k - 1], self.L_inf[k - 1], V)).astype(H)
            self.D_mod[k] = Dm
            self._store_inverse(k, Dm)

    def _store_inverse(self, k, Dm):
        """
        Form the inverse of a modified diagonal block and store it scaled.

        The inverse is computed at self.inv_dtype, rescaled by an exact power
        of two, and only then cast to float16. Applying the scale before the
        cast is what allows an inverse whose natural magnitude lies outside the
        half-precision range to be represented at all.

        Raises FloatingPointError if the inverse is not finite, which indicates
        a modified diagonal block that is singular to working precision, or if
        the scaled inverse still overflows float16.
        """
        if self.inv_dtype == np.dtype(np.float16):
            Y = inv_fp16(Dm).astype(np.float32)
        else:
            Y = sla.inv(Dm.astype(self.inv_dtype))
        if not np.all(np.isfinite(Y)):
            raise FloatingPointError(
                f"non-finite explicit inverse at block {k}: the modified "
                f"diagonal block is singular to working precision"
            )
        t = _pow2_scale(float(np.abs(Y).max()))
        G = (Y * t).astype(H)
        if not np.all(np.isfinite(G)):
            raise FloatingPointError(f"fp16 overflow storing block inverse {k}")
        self.G[k], self.t[k], self.G_inf[k] = G, t, _inf_norm(G)

    def _apply_ginv(self, k, X):
        """
        Apply D_mod_hat[k]^-1 to X in float16 using the stored scaled inverse.

        The product G[k] X equals t[k] times the wanted quantity and routinely
        overflows float16 on its own, so the guard scale q and the factor
        1/t[k] are removed together in a single float32 multiplication rather
        than one after the other; performing them in sequence would materialize
        an intermediate that overflows. On a tensor-core backend this fold
        costs nothing, since the accumulator is float32 in any case.
        """
        P, q = _matmul16_scaled(self.G[k], self.G_inf[k], X)
        return _unscale16(P, q * self.t[k])

    def _solve_one(self, b):
        """
        Two-sweep block solve for a single right-hand side, consisting only of
        matrix products.
        """
        rs = self._rhs_scale(b)
        if rs is None:
            return np.zeros_like(b)
        bb = self._split_embed(b, rs)
        # Forward sweep: substitution replaced by multiplication by the inverse.
        for k in range(1, self.N):
            w = self._apply_ginv(k - 1, bb[k - 1])
            bb[k] = (bb[k] - _matmul16(self.L[k - 1], self.L_inf[k - 1], w)).astype(H)
        # Backward sweep: also entirely matrix products.
        x = [None] * self.N
        x[-1] = self._apply_ginv(self.N - 1, bb[-1])
        for k in range(self.N - 2, -1, -1):
            x[k] = self._apply_ginv(
                k, (bb[k] - _matmul16(self.U[k], self.U_inf[k], x[k + 1])).astype(H)
            )
        return self._finish(x, rs)

    def get_LUP(self):
        """None: no LU factors are formed. See get_inverses()."""
        return None

    def get_inverses(self):
        """
        (G, t, D_mod, L, U), all embedded-real float16 at block size 2m except
        t, which is float64.

        G[k] / t[k] is the inverse of the s-scaled embedded D_mod[k], so
        reconstruction requires both self.s and t.
        """
        return (
            self._stack(self.G, H),
            self.t.copy(),
            self._stack(self.D_mod, H),
            self._stack(self.L, H),
            self._stack(self.U, H),
        )

    def factor_nbytes(self):
        # D_mod is excluded, as in BlockThomasExplicitInv: it is retained for
        # the analysis scripts and is not needed in order to solve.
        G, t, _Dmod, Lb, Ub = self.get_inverses()
        return int(_nbytes(G) + t.nbytes + _nbytes(Lb) + _nbytes(Ub))


# ===========================================================================
# 4) UMFPACK  (CPU, double precision ONLY)
# ===========================================================================
class UMFPACK:
    """
    Sparse LU through scikit-umfpack.

    Installation: conda install -c conda-forge scikit-umfpack

    Precision
    ---------
    UMFPACK supports double precision only; there is no complex64 build.
    Passing a single-precision dtype raises TypeError rather than upcasting
    silently, so that a single-precision batch run records a skip instead of a
    misattributed double-precision result.

    Factor convention
    -----------------
    Explicit factors are available. UMFPACK additionally scales the rows, so
    the reconstruction is

        Pr diag(1/R) A Pc == L U,

    with Pr built from argsort(perm_r), Pc built directly from perm_c, and R
    the row-scaling vector. The convention was determined empirically and
    verified to machine precision; factor_io.diagnose_lu_convention performs
    that determination.
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
        # scikit-umfpack's splu.solve accepts a single right-hand side only, so
        # a 2-D b is solved column by column.
        return _solve_columns(self._solve_one, b)

    def get_LUP(self):
        """(L, U, perm_r, perm_c) with Pr diag(1/R) A Pc == L U."""
        return self.lu.L, self.lu.U, self.lu.perm_r, self.lu.perm_c

    def get_scaling(self):
        """Row-scaling vector R and the do_recip flag, None where absent."""
        return getattr(self.lu, "R", None), getattr(self.lu, "do_recip", None)

    def factor_nbytes(self):
        L, U = self.lu.L, self.lu.U
        nbytes = int(
            L.data.nbytes
            + L.indices.nbytes
            + L.indptr.nbytes
            + U.data.nbytes
            + U.indices.nbytes
            + U.indptr.nbytes
            + np.asarray(self.lu.perm_r).nbytes
            + np.asarray(self.lu.perm_c).nbytes
        )
        R, _ = self.get_scaling()
        if R is not None:
            nbytes += np.asarray(R).nbytes
        return nbytes


# ===========================================================================
# 5) MUMPS  (CPU)
# ===========================================================================
class MUMPS:
    """
    Sparse multifrontal LU through python-mumps.

    Installation: conda install -c conda-forge python-mumps

    The dtype of the matrix selects MUMPS's internal precision, so both
    complex128 and complex64 are available.

    No explicit factors
    -------------------
    MUMPS holds L and U in Fortran-side, possibly MPI-distributed structures,
    and python-mumps's Context API exposes no equivalent of .L, .U or the
    permutations, so get_LUP() returns None. MUMPS does provide a native
    save and restore of a factorization through JOB=7 and JOB=8, but
    python-mumps does not wrap it; in practice, persisting a MUMPS
    factorization means recomputing it.
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
            return self.ctx.solve(b)  # some versions accept 2-D b
        except Exception:
            return _solve_columns(self.ctx.solve, b)

    def get_LUP(self):
        """None: the factors are not exposed by python-mumps."""
        return None

    def factor_nbytes(self):
        # MUMPS reports the total number of factor entries in INFOG(3); a
        # negative value denotes millions of entries. Whether the array is
        # reachable depends on the python-mumps version, so a failure returns 0
        # rather than raising.
        try:
            infog = self.ctx.mumps_instance.infog
            n_entries = int(infog[2])  # INFOG(3), zero-based index 2
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
    GMRES on the GPU through cupyx.scipy.sparse.linalg.gmres.

    Installation: pip install cupy-cuda12x, or cupy-cuda13x.

    Raises RuntimeError at construction when no CUDA device is visible, so a
    CPU-only machine records a skip rather than a failure.

    There is no preconditioner: cupyx provides no incomplete-LU factorization.
    The construction step is therefore only the host-to-device transfer of A,
    and the reported factorization time is the transfer time. Comparisons of
    factorization time against the direct solvers are not meaningful for this
    entry; the solve time and the iteration count are.
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
        # The transfer is asynchronous; synchronize so the reported time covers
        # it rather than only its enqueueing.
        cp.cuda.Stream.null.synchronize()
        self.last_iters = None
        self.last_info = None

    def _solve_one_gpu(self, b_gpu):
        from cupyx.scipy.sparse.linalg import gmres

        iters = [0]

        def cb(x):
            iters[0] += 1

        x_gpu, info = gmres(
            self.A_gpu,
            b_gpu,
            rtol=self.rtol,
            atol=0.0,
            restart=self.restart,
            maxiter=self.maxiter,
            callback=cb,
            callback_type="pr_norm",
        )
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
        # Synchronize before the caller stops its timer: GPU work is enqueued
        # asynchronously and would otherwise not be included.
        cp.cuda.Stream.null.synchronize()
        return cp.asnumpy(x_gpu)

    def get_LUP(self):
        """None: iterative, so no factors exist."""
        return None

    def factor_nbytes(self):
        """Zero: nothing is stored beyond A itself, which is not a factor."""
        return 0


# ===========================================================================
# 7) cuDSS, GPU sparse direct solver, through nvmath-python's DirectSolver.
# ===========================================================================
class CuDSS:
    """
    Sparse direct solve on the GPU through nvmath.sparse.advanced.DirectSolver.

    Installation: pip install nvmath-python[cu12], or [cu13].

    Raises RuntimeError at construction when no CUDA device is visible.

    No explicit factor values
    -------------------------
    cuDSS holds the numerical L and U in opaque device-resident structures, in
    the same category as MUMPS. What is exposed, and what get_metadata()
    returns, is the reordering permutations and the factor nonzero count. Use
    SuperLU or UMFPACK where reconstructable factors are required.

    Right-hand-side binding
    -----------------------
    DirectSolver binds the right-hand side at construction. A later change is
    attempted through solver.reset_operands(b=...); where that is unavailable,
    solve() falls back to rebuilding and refactorizing with the new
    right-hand side. The result is correct, but the separation between
    factorization and solve time is lost for that call, and a notice is
    printed.

    The attribute name row_permutation has not been independently confirmed
    against the nvmath API; get_metadata() reads it through getattr and omits
    entries that are absent.
    """

    def __init__(self, A, dtype=None, nrhs=1):
        if not gpu_available():
            raise RuntimeError(
                "CuDSS requires an NVIDIA GPU with a working CUDA install."
            )
        import nvmath

        self.dtype = np.dtype(dtype) if dtype is not None else A.dtype
        # DirectSolver requires CSR; the shared matrices are stored CSC.
        self.A_csr = A.astype(self.dtype).tocsr()
        self.n = self.A_csr.shape[0]
        self.nrhs = int(nrhs)

        b0 = np.zeros((self.n, self.nrhs), dtype=self.dtype, order="F")
        self.solver = nvmath.sparse.advanced.DirectSolver(self.A_csr, b0)

        t0 = time.perf_counter()
        self.plan_info = self.solver.plan()  # reordering and symbolic phase
        self.plan_seconds = time.perf_counter() - t0

        t0 = time.perf_counter()
        self.fac_info = self.solver.factorize()  # numerical factorization
        self.factor_seconds = time.perf_counter() - t0

    def solve(self, b):
        b = np.asarray(b, dtype=self.dtype, order="F")
        try:
            self.solver.reset_operands(b=b)
            return np.asarray(self.solver.solve())
        except (AttributeError, TypeError) as e:
            print(
                f"CuDSS: reset_operands unavailable ({e}); rebuilding the "
                f"solver with the new right-hand side. The separation of "
                f"factorization and solve time is lost for this call."
            )
            import nvmath

            self.free()
            self.solver = nvmath.sparse.advanced.DirectSolver(self.A_csr, b)
            self.plan_info = self.solver.plan()
            self.fac_info = self.solver.factorize()
            return np.asarray(self.solver.solve())

    def get_LUP(self):
        """None: the factor values are not exposed. See get_metadata()."""
        return None

    def get_metadata(self):
        """Reordering permutations and factor nonzero count, where exposed."""
        md = {
            "col_permutation": getattr(self.plan_info, "col_permutation", None),
            "row_permutation": getattr(self.plan_info, "row_permutation", None),
            "lu_nnz": getattr(self.fac_info, "lu_nnz", None),
        }
        return {k: v for k, v in md.items() if v is not None}

    def factor_nbytes(self):
        # Estimate of the factor values only, from the reported nonzero count.
        # The index arrays are internal and their size is unknown. Returns 0
        # when lu_nnz is not exposed by this version.
        lu_nnz = getattr(self.fac_info, "lu_nnz", None)
        return int(lu_nnz * self.dtype.itemsize) if lu_nnz is not None else 0

    def free(self):
        """Release the device-side solver. Idempotent and never raises."""
        try:
            self.solver.free()
        except Exception:
            pass
