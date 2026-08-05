"""
solver_classes.py -- importable solver classes for the benchmark notebook.

Put this file in /solvers, then from the notebook (which lives in /jupyter):

    import sys
    from pathlib import Path
    sys.path.append(str((Path.cwd() / ".." / "solvers").resolve()))
    from solver_classes import (
        SparseLU, GMRES, extract_blocks_sparse,                # CPU, always available
        BlockThomas, BlockThomasExplicitInv,                   # Higham Alg. 13.3, impl 1 / 2
        BlockThomasFP16, BlockThomasExplicitInvFP16,           # the same two in fp16
        find_block_slices, block_sizes_from_matrix, offband_nnz,   # custom partitions
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
    BlockThomasFP16      : ditto, embedded-real fp16 at block size 2*bs
    BlockThomasExplicitInv,
    BlockThomasExplicitInvFP16
                         : per-block EXPLICIT INVERSES, no LU factors (Higham
                           Alg. 13.3, Implementation 2) -- get_LUP() returns None,
                           use get_inverses() instead.
    MUMPS                : nothing -- factors live in internal Fortran-side structures;
               python-mumps's Context API does not expose them.
    cuDSS    : permutations + factor nnz count only -- factor VALUES are
               opaque GPU-resident structures, not exposed by the public API.

BLOCK PARTITIONS. All four Block Thomas variants infer their partition from
the shapes of the D blocks, so a custom non-uniform partition enters at one
place only -- extract_blocks_sparse(A, block_sizes), which takes an int
(uniform) or a sequence of sizes. find_block_slices() derives such a sequence
from the sparsity pattern; check it with offband_nnz(A, sizes) == 0 first.
For a custom partition get_LUP()/get_inverses() return LISTS of per-block
arrays instead of stacked (N, bs, bs) arrays, since the blocks are ragged;
uniform runs keep the stacked layout the analysis scripts already read.
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


def _nbytes(part):
    """Byte count of a factor part that may be a stacked array (uniform
    partition) or a list of per-block arrays (custom partition)."""
    if isinstance(part, list):
        return int(sum(np.asarray(a).nbytes for a in part))
    return int(np.asarray(part).nbytes)


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
# BLOCK STRUCTURE -- uniform and custom (non-uniform) partitions
#
# Every Block Thomas variant below infers its partition from the shapes of
# the D blocks it is handed, so a non-uniform partition enters the pipeline
# at ONE place only: extract_blocks_sparse(A, block_sizes). Pass an int for
# the historical uniform behaviour, or a sequence of per-block sizes summing
# to n for a custom partition.
#
# find_block_slices() derives such a partition straight from the sparsity
# pattern -- it is determine_custom_block_size.py, folded in here so the
# detector and the solvers consuming its output live in one module.
# ===========================================================================
def find_block_slices(a):
    """
    Variable-size block structure of a sparse CSR matrix, found by growing a
    reach "frontier" row by row. A block boundary is declared once the
    frontier stops advancing -- i.e. once no row in the current unresolved
    range points to a column beyond what has already been reached.

    Requires a.indices sorted ascending per row; sort_indices() is called
    here so callers do not have to.

    NOTE: the frontier only looks FORWARD (largest column index per row), so
    the partition it returns is guaranteed block-tridiagonal only when A is
    structurally symmetric. Always confirm with offband_nnz(A, sizes) == 0
    before handing the partition to a Block Thomas solver.
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

    # the first slice is always the single seed row -- merge it into the second
    if len(block_slices) > 1:
        block_slices[1] = slice(block_slices[0].start, block_slices[1].stop)
        block_slices.pop(0)

    return block_slices


def block_sizes_from_matrix(A):
    """find_block_slices() as a plain tuple of block sizes."""
    return tuple(s.stop - s.start for s in find_block_slices(A))


def normalize_block_sizes(n, block_sizes):
    """
    Accept either an int (uniform) or a sequence of per-block sizes, and
    return a validated tuple summing to n. Raises on a partition that does
    not tile the matrix -- silently truncating would produce a wrong answer
    rather than an error.
    """
    if np.isscalar(block_sizes):
        bs = int(block_sizes)
        if n % bs:
            raise ValueError(f"n={n} is not divisible by block size {bs}; "
                             f"pass an explicit list of block sizes instead")
        return (bs,) * (n // bs)
    sizes = tuple(int(s) for s in block_sizes)
    if any(s <= 0 for s in sizes):
        raise ValueError(f"all block sizes must be positive, got {sizes}")
    if sum(sizes) != n:
        raise ValueError(f"block sizes sum to {sum(sizes)}, expected n={n}")
    return sizes


def block_offsets(block_sizes):
    """Cumulative block boundaries, length N+1 (offsets[k]:offsets[k+1])."""
    return np.concatenate([[0], np.cumsum(np.asarray(block_sizes, dtype=np.int64))])


def offband_nnz(A, block_sizes):
    """
    Number of stored entries lying OUTSIDE the block-tridiagonal band of the
    given partition. 0 means A really is block tridiagonal for this
    partition, which is the precondition every Block Thomas variant assumes;
    anything else means the blocks discard real couplings and the solve will
    be silently wrong.
    """
    sizes = normalize_block_sizes(A.shape[0], block_sizes)
    off = block_offsets(sizes)
    C = A.tocoo()
    bi = np.searchsorted(off, C.row, side="right") - 1
    bj = np.searchsorted(off, C.col, side="right") - 1
    return int(np.count_nonzero(np.abs(bi - bj) > 1))


def extract_blocks_sparse(As, block_sizes):
    """
    Block-tridiagonal blocks from a sparse matrix (densifies only the small
    blocks). `block_sizes` is an int (uniform) or a sequence summing to n.

    For a non-uniform partition the off-diagonal blocks are RECTANGULAR:
    L[k] is (n[k+1], n[k]) and U[k] is (n[k], n[k+1]).
    """
    n = As.shape[0]
    sizes = normalize_block_sizes(n, block_sizes)
    off = block_offsets(sizes)
    N = len(sizes)
    Ac = As.tocsr()
    D = [np.asarray(Ac[off[k]:off[k + 1], off[k]:off[k + 1]].todense())
         for k in range(N)]
    L = [np.asarray(Ac[off[k + 1]:off[k + 2], off[k]:off[k + 1]].todense())
         for k in range(N - 1)]
    U = [np.asarray(Ac[off[k]:off[k + 1], off[k + 1]:off[k + 2]].todense())
         for k in range(N - 1)]
    return D, L, U


class _BlockThomasBase:
    """
    Block bookkeeping shared by all four Block Thomas variants: partition
    inference, shape validation, and splitting/joining a flat RHS.

    The partition is inferred from the D blocks, so no variant takes a
    block-size argument -- pass blocks from extract_blocks_sparse() and the
    solver follows whatever partition those blocks encode.
    """

    def _init_blocks(self, D, L, U):
        if len(L) != len(D) - 1 or len(U) != len(D) - 1:
            raise ValueError(f"expected {len(D) - 1} off-diagonal blocks, "
                             f"got len(L)={len(L)}, len(U)={len(U)}")
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
        # `bs` stays an int for uniform partitions (what factor_io and the
        # older notebooks expect); None signals a ragged partition.
        self.bs = self.block_sizes[0] if self.uniform else None

    def _split(self, b):
        """Flat (n,) or (n, nrhs) RHS -> list of per-block views."""
        off = self.offsets
        return [b[off[k]:off[k + 1]] for k in range(self.N)]

    def _stack(self, blocks, dtype):
        """
        Stack per-block arrays into one (N, ...) array when the partition is
        uniform, else return the list unchanged. Uniform runs therefore keep
        the exact on-disk factor layout the analysis scripts already read.
        """
        if not blocks:
            return np.empty((0, 0, 0), dtype)
        return np.stack(blocks) if self.uniform else list(blocks)


# ===========================================================================
# 3) BLOCK THOMAS -- from the notebook
#    Higham, "Accuracy and Stability of Numerical Algorithms", Alg. 13.3,
#    IMPLEMENTATION 1: diagonal blocks factorized by GEPP (LAPACK getrf via
#    scipy.linalg.lu_factor); every place the algorithm needs D_mod_k^{-1},
#    it instead does a triangular substitution against the stored LU factors
#    (scipy.linalg.lu_solve). No block is ever explicitly inverted.
# ===========================================================================
class BlockThomas(_BlockThomasBase):
    def __init__(self, D, L, U, dtype=None):
        self._init_blocks(D, L, U)
        self.dtype = np.dtype(dtype) if dtype is not None else D[0].dtype
        D = [d.astype(self.dtype) for d in D]
        self.L = [l.astype(self.dtype) for l in L]
        self.U = [u.astype(self.dtype) for u in U]
        self.D_mod  = [None] * self.N
        self.lu_piv = [None] * self.N
        self.D_mod[0]  = D[0]
        self.lu_piv[0] = sla.lu_factor(D[0])
        for k in range(1, self.N):
            self.D_mod[k]  = D[k] - self.L[k-1] @ sla.lu_solve(self.lu_piv[k-1], self.U[k-1])
            self.lu_piv[k] = sla.lu_factor(self.D_mod[k])

    def solve(self, b):
        was_array = not isinstance(b, list)
        bb = self._split(b) if was_array else [bk.copy() for bk in b]
        bb = [bk.astype(self.dtype) for bk in bb]
        for k in range(1, self.N):
            bb[k] = bb[k] - self.L[k-1] @ sla.lu_solve(self.lu_piv[k-1], bb[k-1])
        x = [None] * self.N
        x[-1] = sla.lu_solve(self.lu_piv[-1], bb[-1])
        for k in range(self.N-2, -1, -1):
            x[k] = sla.lu_solve(self.lu_piv[k], bb[k] - self.U[k] @ x[k+1])
        return np.concatenate(x, axis=0) if was_array else x

    def get_LUP(self):
        """(L, U, Dmod_lu, Dmod_piv). Stacked (N, bs, bs) arrays for a uniform
        partition; lists of per-block arrays for a custom one."""
        Lb   = self._stack(self.L, self.dtype)
        Ub   = self._stack(self.U, self.dtype)
        Dlu  = self._stack([lu  for lu, piv in self.lu_piv], self.dtype)
        Dpiv = self._stack([piv for lu, piv in self.lu_piv], np.int32)
        return Lb, Ub, Dlu, Dpiv

    def factor_nbytes(self):
        return int(sum(_nbytes(part) for part in self.get_LUP()))


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
class BlockThomasExplicitInv(_BlockThomasBase):
    def __init__(self, D, L, U, dtype=None):
        self._init_blocks(D, L, U)
        self.dtype = np.dtype(dtype) if dtype is not None else D[0].dtype
        D = [d.astype(self.dtype) for d in D]
        self.L = [l.astype(self.dtype) for l in L]
        self.U = [u.astype(self.dtype) for u in U]
        self.D_mod     = [None] * self.N
        self.D_mod_inv = [None] * self.N
        self.D_mod[0]     = D[0]
        self.D_mod_inv[0] = sla.inv(D[0])
        for k in range(1, self.N):
            self.D_mod[k]     = D[k] - self.L[k-1] @ self.D_mod_inv[k-1] @ self.U[k-1]
            self.D_mod_inv[k] = sla.inv(self.D_mod[k])

    def solve(self, b):
        was_array = not isinstance(b, list)
        bb = self._split(b) if was_array else [bk.copy() for bk in b]
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
        """Explicit block inverses D_mod_inv[k] plus the off-diagonal blocks.
        The Implementation-2 analogue of get_LUP() -- there are no L/U/pivot
        factors, only the explicitly formed inverses. Stacked (N, bs, bs)
        arrays for a uniform partition, lists for a custom one.

        D_mod is returned as well: it is what the growth-factor analysis needs
        to assemble the global U (Implementation 2 never forms LU factors, but
        the same global L, U exist mathematically and D_mod is their diagonal)."""
        Db_inv = self._stack(self.D_mod_inv, self.dtype)
        Db     = self._stack(self.D_mod, self.dtype)
        Lb     = self._stack(self.L, self.dtype)
        Ub     = self._stack(self.U, self.dtype)
        return Db_inv, Db, Lb, Ub

    def factor_nbytes(self):
        # D_mod is bookkeeping for the analysis scripts, not part of the
        # factorization Implementation 2 actually stores -- excluded here so
        # the memory figure stays comparable with Implementation 1.
        Db_inv, _Db, Lb, Ub = self.get_inverses()
        return int(_nbytes(Db_inv) + _nbytes(Lb) + _nbytes(Ub))


# ===========================================================================
# 3c) HALF PRECISION (fp16) BLOCK THOMAS -- both implementations
#
# Complex blocks are handled by the exact real embedding
#       z = a + bi  ->  [[a, -b], [b, a]]
# so every operation is real fp16 and no complex32 dtype is needed. A complex
# m x m block becomes a real 2m x 2m block; a rectangular (p, q) coupling
# block becomes (2p, 2q), so custom non-uniform partitions work unchanged.
#
# fp16 backend is numpy float16: numpy evaluates each op in fp32 internally
# and rounds the result back to fp16, i.e. chop-style per-op rounding with
# the same accumulate semantics as tensor cores. A torch/GPU backend is a
# mechanical swap.
#
# SCALING. fp16 normals span only [6.1e-5, 65504], so raw QTBM blocks have to
# be brought into range first. Both classes apply one global power-of-two
# scale s (exact, no rounding error) chosen so the largest entry of the
# embedded A lands near 1024, leaving ~64x headroom before overflow.
# ===========================================================================

H = np.float16
_FP16_TARGET = 1024.0          # where the largest scaled entry is aimed


def embed_block(Z):
    """Complex (p, q) -> real (2p, 2q), exactly."""
    a, b = np.ascontiguousarray(Z.real), np.ascontiguousarray(Z.imag)
    return np.block([[a, -b], [b, a]])


def embed_vec(z):
    return np.concatenate([z.real, z.imag], axis=0)


def unembed_vec(v):
    m = v.shape[0] // 2
    return v[:m] + 1j * v[m:]


_FP16_PROD_TARGET = 16384.0    # where a guarded matmul result is aimed (4x headroom)


def _pow2_scale(amax, target=_FP16_TARGET):
    """Largest power of two t with amax * t <= target. Exact in binary fp."""
    if not np.isfinite(amax) or amax == 0.0:
        raise FloatingPointError(f"cannot scale a block with max |entry| = {amax}")
    return 2.0 ** np.floor(np.log2(target / amax))


def _inf_norm(A):
    """Max absolute row sum -- the exact bound |(A @ X)_ij| <= ||A||_inf * max|X|."""
    return float(np.abs(A.astype(np.float32)).sum(axis=1).max())


def _matmul16_scaled(Ah, a_inf, X):
    """
    fp16 matrix product, guarded against intermediate overflow. Returns
    (P, q) with P ~ q * (Ah @ X) in fp16 and q an exact power of two.

    A dense fp16 matmul of two arrays whose entries sit near the top of the
    scaled range overflows purely from the accumulation length: with entries
    at ~1e3 and a block dimension of ~800 the exact result is ~1e9, far past
    the fp16 ceiling of 65504, even though the quantity the algorithm actually
    wants is O(1). This bites at the block sizes the QTBM materials use
    (bs=416 embeds to an 832-long accumulation).

    The guard rescales X by q, chosen from the rigorous bound
        |(Ah @ X)_ij| <= ||Ah||_inf * max|X|
    so the product cannot leave fp16 range. Powers of two are exact in binary
    floating point, so this changes no digits of the result -- it only moves
    where the computation sits in the exponent range. `a_inf` is ||Ah||_inf,
    precomputed once per stored block.

    q is returned rather than undone in place so a caller that owes another
    power-of-two factor (Implementation 2 divides by its per-block inverse
    scale t) can fold both into a single multiply. Undoing them in sequence
    would materialize an intermediate that overflows on its own.
    """
    xmax = float(np.abs(X).max())
    if xmax == 0.0 or a_inf == 0.0:
        return np.zeros((Ah.shape[0],) + X.shape[1:], dtype=H), 1.0
    q = _pow2_scale(a_inf * xmax, target=_FP16_PROD_TARGET)
    Xq = (X.astype(np.float32) * np.float32(q)).astype(H)
    return (Ah @ Xq).astype(H), q


def _unscale16(P, q):
    """Undo a power-of-two scale in fp32, then round back to fp16. Done in
    fp32 because 1/q is itself frequently subnormal (or infinite) in fp16."""
    return (P.astype(np.float32) * np.float32(1.0 / q)).astype(H)


def _matmul16(Ah, a_inf, X):
    """Guarded fp16 Ah @ X with the guard scale undone -- see _matmul16_scaled."""
    return _unscale16(*_matmul16_scaled(Ah, a_inf, X))


def lu_fp16(Dm):
    """Packed LU with partial pivoting of a (k, k) fp16 array, fp16 rounding
    after every operation. Returns (LU, piv)."""
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
        LU[j + 1:, j] = (LU[j + 1:, j] / LU[j, j]).astype(H)
        LU[j + 1:, j + 1:] = (LU[j + 1:, j + 1:]
                              - np.outer(LU[j + 1:, j], LU[j, j + 1:])).astype(H)
    return LU, piv


def lu_solve_fp16(LU, piv, Rhs):
    """Solve (LU) X = Rhs entirely in fp16. Rhs is (k,) or (k, r)."""
    X = Rhs.astype(H)[piv].copy()
    one_d = X.ndim == 1
    if one_d:
        X = X[:, None]
    k = LU.shape[0]
    for j in range(k):                       # unit-L forward substitution
        X[j + 1:] = (X[j + 1:] - np.outer(LU[j + 1:, j], X[j])).astype(H)
    for j in range(k - 1, -1, -1):           # U backward substitution
        X[j] = (X[j] / LU[j, j]).astype(H)
        X[:j] = (X[:j] - np.outer(LU[:j, j], X[j])).astype(H)
    return X[:, 0] if one_d else X


def inv_fp16(Dm):
    """Explicit inverse of a (k, k) fp16 block, computed entirely in fp16."""
    return lu_solve_fp16(*lu_fp16(Dm), np.eye(Dm.shape[0], dtype=H))


class _BlockThomasFP16Base(_BlockThomasBase):
    """Shared fp16 setup: embedding, global power-of-two scaling, RHS handling."""

    def _init_fp16(self, D, L, U):
        self._init_blocks(D, L, U)
        self.dtype = H
        # Scale from D, L AND U: the off-diagonal coupling blocks can carry
        # the largest entries, and scaling on D alone would let them overflow.
        amax = max(max(float(np.abs(b).max()) for b in D),
                   max((float(np.abs(b).max()) for b in L), default=0.0),
                   max((float(np.abs(b).max()) for b in U), default=0.0))
        self.s = _pow2_scale(amax)
        to16 = lambda X: (embed_block(X) * self.s).astype(H)
        self.L = [to16(l) for l in L]
        self.U = [to16(u) for u in U]
        # row-sum norms for the overflow guard in _matmul16, computed once
        self.L_inf = [_inf_norm(l) for l in self.L]
        self.U_inf = [_inf_norm(u) for u in self.U]
        return to16

    def _split_embed(self, b, rs):
        """Per-block embedded fp16 RHS, scaled by rs."""
        return [embed_vec(bk * rs).astype(H) for bk in self._split(b)]

    def solve(self, b):
        b = np.asarray(b)
        if b.ndim == 1:
            return self._solve_one(b)
        return np.column_stack([self._solve_one(np.ascontiguousarray(b[:, j]))
                                for j in range(b.shape[1])])

    def _rhs_scale(self, b):
        """Power-of-two RHS scale bringing b to O(1) -- fp16 underflows below
        ~6e-5, and QTBM right-hand sides are routinely far smaller."""
        rmax = float(np.abs(b).max())
        return None if rmax == 0 else _pow2_scale(rmax, target=1.0)

    def _finish(self, x_blocks, rs):
        """Unembed, promote to fp64, undo both scalings.
        Factors approximate (s*A)^{-1} = A^{-1}/s and the RHS carried rs."""
        out = np.concatenate([unembed_vec(xk.astype(np.float64)) for xk in x_blocks])
        return out * (self.s / rs)


class BlockThomasFP16(_BlockThomasFP16Base):
    """Implementation 1 (LU + substitution, no explicit inverses) in fp16.

    D, L, U are lists of complex blocks -- the same inputs BlockThomas takes,
    from extract_blocks_sparse, uniform or custom partition.
    """

    def __init__(self, D, L, U, dtype=None):    # dtype ignored (always fp16)
        to16 = self._init_fp16(D, L, U)
        self.lu_piv = [None] * self.N
        self.D_mod = [None] * self.N
        Dm = to16(D[0])
        self.D_mod[0] = Dm
        self.lu_piv[0] = lu_fp16(Dm)
        for k in range(1, self.N):
            # D_mod[k] = D[k] - L[k-1] @ D_mod[k-1]^-1 @ U[k-1], via lu_solve
            W = lu_solve_fp16(*self.lu_piv[k - 1], self.U[k - 1])
            Dm = (to16(D[k]) - _matmul16(self.L[k - 1], self.L_inf[k - 1], W)).astype(H)
            self.D_mod[k] = Dm
            self.lu_piv[k] = lu_fp16(Dm)

    def _solve_one(self, b):
        rs = self._rhs_scale(b)
        if rs is None:
            return np.zeros_like(b)
        bb = self._split_embed(b, rs)
        for k in range(1, self.N):                       # forward sweep
            w = lu_solve_fp16(*self.lu_piv[k - 1], bb[k - 1])
            bb[k] = (bb[k] - _matmul16(self.L[k - 1], self.L_inf[k - 1], w)).astype(H)
        x = [None] * self.N
        x[-1] = lu_solve_fp16(*self.lu_piv[-1], bb[-1])
        for k in range(self.N - 2, -1, -1):              # block back substitution
            x[k] = lu_solve_fp16(
                *self.lu_piv[k],
                (bb[k] - _matmul16(self.U[k], self.U_inf[k], x[k + 1])).astype(H))
        return self._finish(x, rs)

    def get_LUP(self):
        """Embedded-real fp16 factors at block size 2*bs. Reconstruction needs
        self.s: these are factors of s * A_embedded, not of A."""
        Lb   = self._stack(self.L, H)
        Ub   = self._stack(self.U, H)
        Dlu  = self._stack([lu  for lu, piv in self.lu_piv], H)
        Dpiv = self._stack([piv for lu, piv in self.lu_piv], np.int32)
        return Lb, Ub, Dlu, Dpiv

    def factor_nbytes(self):
        return int(sum(_nbytes(part) for part in self.get_LUP()))


class BlockThomasExplicitInvFP16(_BlockThomasFP16Base):
    """Implementation 2 (explicit inverses, matmul-only solve) in fp16.

    Storage and application are fp16 throughout -- that is the point of the
    method: every solve stage is a dense matmul, which is what tensor cores
    want. `inv_dtype` controls the precision in which the inverse itself is
    FORMED before being rounded down to fp16 for storage:

        inv_dtype=np.float32 (default)
            Each modified diagonal block is promoted to fp32, inverted with
            LAPACK, then rounded to fp16. This is the standard mixed-precision
            split (factor high, store/apply low): explicit inversion is the
            single least stable step in the algorithm, and doing it at fp16
            costs accuracy that no amount of care in the sweeps recovers,
            while the fp32 inversion is O(N) small dense inversions and does
            not change the memory footprint of what is stored.

        inv_dtype=np.float16
            Inverse formed in fp16 too (lu_fp16 + substitution against I), so
            the whole factorization is genuinely half precision. Slower and
            markedly less accurate -- provided so the cost of the fp32
            inversion can be measured rather than assumed.

    PER-BLOCK RESCALING. Implementation 1 needs one global scale because it
    only ever stores quantities of the same magnitude as A. Implementation 2
    stores INVERSES, whose entries scale like 1/|D_mod|, i.e. in the opposite
    direction: the same global scale that keeps A inside fp16 range drives the
    inverses toward the fp16 underflow threshold. Each inverse therefore gets
    its own power-of-two scale t[k], stored alongside it:

        G[k] = t[k] * D_mod_hat[k]^-1        (fp16)

    with t[k] exact in binary floating point, so rescaling introduces no
    rounding error of its own. The recursion stays scale-free in the hatted
    quantities (D_hat = s*D, L_hat = s*L, U_hat = s*U):

        D_mod_hat[k] = D_hat[k] - (1/t[k-1]) * L_hat[k-1] @ G[k-1] @ U_hat[k-1]
    """

    def __init__(self, D, L, U, dtype=None, inv_dtype=np.float32):
        to16 = self._init_fp16(D, L, U)
        self.inv_dtype = np.dtype(inv_dtype)
        if self.inv_dtype not in (np.dtype(np.float16), np.dtype(np.float32),
                                  np.dtype(np.float64)):
            raise ValueError(f"inv_dtype must be float16/32/64, got {self.inv_dtype}")

        self.G = [None] * self.N           # scaled explicit inverses, fp16
        self.t = np.ones(self.N)           # their power-of-two scales
        self.G_inf = [0.0] * self.N        # row-sum norms for the matmul guard
        self.D_mod = [None] * self.N       # kept for the growth-factor analysis

        Dm = to16(D[0])
        self.D_mod[0] = Dm
        self._store_inverse(0, Dm)
        for k in range(1, self.N):
            V = self._apply_ginv(k - 1, self.U[k - 1])
            Dm = (to16(D[k]) - _matmul16(self.L[k - 1], self.L_inf[k - 1], V)).astype(H)
            self.D_mod[k] = Dm
            self._store_inverse(k, Dm)

    def _store_inverse(self, k, Dm):
        """Form inv(Dm) at inv_dtype, rescale by an exact power of two, and
        store it in fp16. The scaling is applied before the fp16 cast, so an
        inverse that would overflow (or underflow) fp16 in its natural
        scaling still round-trips."""
        if self.inv_dtype == np.dtype(np.float16):
            Y = inv_fp16(Dm).astype(np.float32)
        else:
            Y = sla.inv(Dm.astype(self.inv_dtype))
        if not np.all(np.isfinite(Y)):
            raise FloatingPointError(
                f"non-finite explicit inverse at block {k}: the modified "
                f"diagonal block is singular to working precision")
        t = _pow2_scale(float(np.abs(Y).max()))
        G = (Y * t).astype(H)
        if not np.all(np.isfinite(G)):
            raise FloatingPointError(f"fp16 overflow storing block inverse {k}")
        self.G[k], self.t[k], self.G_inf[k] = G, t, _inf_norm(G)

    def _apply_ginv(self, k, X):
        """fp16  D_mod_hat[k]^-1 @ X, using the stored scaled inverse.

        G[k] @ X is t[k] times the wanted quantity and routinely overflows
        fp16 on its own, so the matmul guard's scale and 1/t[k] are undone
        together in a single fp32 multiply rather than one after the other.
        On a tensor-core backend this fold is free -- the accumulator is fp32
        anyway."""
        P, q = _matmul16_scaled(self.G[k], self.G_inf[k], X)
        return _unscale16(P, q * self.t[k])

    def _solve_one(self, b):
        rs = self._rhs_scale(b)
        if rs is None:
            return np.zeros_like(b)
        bb = self._split_embed(b, rs)
        # forward sweep: substitution replaced by matmul with the inverse
        for k in range(1, self.N):
            w = self._apply_ginv(k - 1, bb[k - 1])
            bb[k] = (bb[k] - _matmul16(self.L[k - 1], self.L_inf[k - 1], w)).astype(H)
        # back substitution: also entirely matmuls
        x = [None] * self.N
        x[-1] = self._apply_ginv(self.N - 1, bb[-1])
        for k in range(self.N - 2, -1, -1):
            x[k] = self._apply_ginv(
                k, (bb[k] - _matmul16(self.U[k], self.U_inf[k], x[k + 1])).astype(H))
        return self._finish(x, rs)

    def get_LUP(self):
        return None                        # no LU factors -- use get_inverses()

    def get_inverses(self):
        """(G, t, D_mod, L, U) -- all embedded-real fp16 at block size 2*bs.
        G[k] * (1/t[k]) is the inverse of the s-scaled embedded D_mod[k], so
        reconstruction needs both self.s and t."""
        return (self._stack(self.G, H), self.t.copy(),
                self._stack(self.D_mod, H),
                self._stack(self.L, H), self._stack(self.U, H))

    def factor_nbytes(self):
        # D_mod excluded, as in BlockThomasExplicitInv: it is analysis
        # bookkeeping, not part of what Implementation 2 stores to solve.
        G, t, _Dmod, Lb, Ub = self.get_inverses()
        return int(_nbytes(G) + t.nbytes + _nbytes(Lb) + _nbytes(Ub))


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