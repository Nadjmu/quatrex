"""
Uniform benchmarking of every solver on one linear system.

Purpose
-------
This module holds the single bench() implementation used by every driver in
the project, so that a solver added or a measurement corrected here propagates
to all of them and no two drivers can measure the same quantity differently.

Input
-----
    As      : the system matrix, sparse.
    B       : the right-hand side, (n,) or (n, nrhs).
    idx     : energy index, used for reporting and for the HDF5 group path.
    bs      : the Block Thomas partition, an int for a uniform partition or a
              sequence of per-block sizes for a custom one.
    dtypes  : working precisions to run, in order. The first defines the
              baseline.
    solvers : which solvers to run.
    exclude : (solver, dtype) combinations to skip deliberately.
    h5file  : an open HDF5 file in mode "a" or "r+"; results are appended to it
              when save is true.

Algorithm
---------
Each solver is constructed once per precision, which performs the entire
factorization and is what the reported factorization time measures, and is then
applied to B, which is what the solve time measures. For each combination the
following are recorded:

    factor                        factorization wall time, seconds
    solve                         solve wall time, seconds
    res                           ||As x - B|| / ||B||, the relative residual
    mem                           solver-reported factor footprint, bytes
    vs_base                       ||x - x_base|| / ||x_base||, where x_base is
                                  the SuperLU solution at the first requested
                                  precision
    backward_error_normwise       eta, Rigal-Gaches normwise backward error
    backward_error_componentwise  omega, Oettli-Prager componentwise backward
                                  error

eta and omega are computed by backward_errors() from the same residual As x - B
that res is; see its docstring for the formulas. Both, along with the residual
vector itself, are also written into every saved HDF5 group, since they are
cheap to compute once per solve and answer a different question than res: they
report the perturbation of the problem (As, B) that x solves exactly, not the
size of the raw residual.

Result keys are "<solver>_<suffix>" in the canonical solver spelling defined by
cli.SOLVERS: superlu_c128, umfpack_c128, block-thomas-inv_c64.

The two half-precision Block Thomas variants are the exception. They are
precision-fixed, so running them inside the precision loop would repeat
identical work; they run once per index outside it, under the unsuffixed keys
block-thomas-fp16 and block-thomas-inv-fp16 and the storage label
"complex32".
They are absent from DEFAULT_SOLVERS because their kernels are written in
NumPy and are orders of magnitude slower than LAPACK: request them explicitly,
through cli.FP16_SOLVERS, when accuracy rather than timing is being measured.

Solver names are the canonical kebab-case forms of cli.SOLVERS. The HDF5 group
each result is written to is resolved through cli.h5_group, so the stored
layout is unchanged.

Skips are per (solver, dtype) and are reported rather than raised: UMFPACK has
no single-precision build, the GPU solvers require a visible CUDA device, and
any solver whose package is absent is skipped. This keeps a batch run over
heterogeneous machines comparable.

Block partitions are validated before any Block Thomas variant runs, unless
check_blocks is false. A partition that cuts a real coupling produces a
plausible but wrong solution without raising, so the check is on by default;
it costs one pass over the nonzeros.

Output
------
A dict of the metrics above, and, when save is true, the corresponding groups
written into h5file by factor_io.

Backwards compatibility: bench(..., dtype=np.complex128) is accepted and
treated as dtypes=(np.complex128,).
"""

import time
import numpy as np
import scipy.sparse.linalg as splinalg

from solver_classes import (
    SparseLU, GMRES, extract_blocks_sparse, offband_nnz,
    BlockThomas, BlockThomasExplicitInv,
    BlockThomasFP16, BlockThomasExplicitInvFP16,
    UMFPACK, MUMPS, GMRESCuPy, CuDSS, gpu_available,
)
import factor_io as fio
from cli import (
    DEFAULT_SOLVERS, FP16_SOLVERS, dtype_suffix, h5_group, label,
)

DEFAULT_DTYPES = (np.complex128, np.complex64)

# Storage label for the half-precision results. Not a NumPy dtype:
# np.dtype("complex32") does not exist.
FP16_LABEL = "complex32"


def _sfx(dt):
    """Short suffix for a precision, used in the result keys."""
    return dtype_suffix(np.dtype(dt).name)


def _timed(fn, *args, **kwargs):
    """Call fn and return (result, elapsed wall time in seconds)."""
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _line(label, t_f, t_s, res, mem, extra=""):
    """One line of the per-index report."""
    print(f"  {label:20s}: factor {t_f*1e3:8.2f} ms  solve {t_s*1e3:8.3f} ms"
          f"  res {res:.1e}  mem {mem/1e6:7.1f} MB{extra}")


NORMWISE_ORDS = (1, 2, np.inf)


def _matrix_norm(A, ord):
    """
    ||A||_p of a sparse matrix, the operator norm induced by the vector
    p-norm, for p in {1, 2, inf}.

    p = 1 and p = inf are the maximum absolute column and row sum
    respectively, exact and O(nnz). p = 2 is the spectral norm, the largest
    singular value, gotten from a sparse SVD (ARPACK via scipy.sparse.linalg
    .svds) rather than a dense one, since As is never densified elsewhere in
    this module. The Frobenius norm is deliberately not offered here: it is
    not the operator norm induced by any vector p-norm, so it does not appear
    in the Rigal-Gaches theorem and would silently give a non-tight, formally
    incorrect backward error.

    Returns None for p = 2 when ARPACK's Lanczos iteration fails to converge,
    which happens on matrices whose top singular value is poorly separated
    from its neighbours -- expected near a band edge, where the system is
    close to singular. The caller skips eta_2 for that index rather than
    treating the failure as fatal.
    """
    if ord == 1:
        Ac = A.tocsc(copy=True)
        Ac.data = np.abs(Ac.data)
        return float(np.asarray(Ac.sum(axis=0)).ravel().max())
    if ord == np.inf:
        Ac = A.tocsr(copy=True)
        Ac.data = np.abs(Ac.data)
        return float(np.asarray(Ac.sum(axis=1)).ravel().max())
    if ord == 2:
        try:
            sigma_max = splinalg.svds(A, k=1, return_singular_vectors=False)
            return float(sigma_max[0])
        except splinalg.ArpackNoConvergence:
            return None
    raise ValueError(f"unsupported norm order {ord!r}; use 1, 2 or np.inf")


def _abs_matvec(A, X):
    """|A| @ |X|, the absolute value taken before the product."""
    Ac = A.tocsr(copy=True)
    Ac.data = np.abs(Ac.data)
    return Ac @ np.abs(X)


def backward_errors(As, X, B, normA, R=None):
    """
    Residual and backward errors of a computed solution X of As X = B.

        eta_p(X)  = ||R||_p / (||As||_p ||X||_p + ||B||_p),   p in {1, 2, inf}
        omega(X)  = max_ij |R_ij| / (|As| |X| + |B|)_ij       (0/0 -> 0)

    eta_p is the normwise backward error of Rigal and Gaches: the smallest
    relative perturbation of As and B, measured in the p-norm, that makes X
    an exact solution. It is computed at p = 1, 2 and inf, since the theorem
    holds for any operator norm and the three differ in which entries of As
    they weight most. omega is the componentwise backward error of Oettli and
    Prager, the same statement with entrywise rather than normwise
    perturbation bounds; unlike eta it has no p-norm family, since the
    entrywise max is already the tightest such bound. See Higham, Accuracy
    and Stability of Numerical Algorithms, Theorems 7.1 and 7.3.

    For several right-hand sides (X, B two-dimensional), every quantity is the
    worst case over the columns. normA is a dict {1: ..., 2: ..., inf: ...},
    passed in rather than recomputed here, since it does not depend on the
    solver, the precision or the column.

    Returns (R, etas, omega): R = As @ X - B, etas a dict keyed like normA,
    omega a float.
    """
    if R is None:
        R = As @ X - B
    X2 = X if X.ndim == 2 else X[:, None]
    B2 = B if B.ndim == 2 else B[:, None]
    R2 = R if R.ndim == 2 else R[:, None]

    etas = {}
    for p in NORMWISE_ORDS:
        if normA[p] is None:                    # svds(p=2) failed to converge
            etas[p] = None
            continue
        normX = np.linalg.norm(X2, ord=p, axis=0)
        normB = np.linalg.norm(B2, ord=p, axis=0)
        normR = np.linalg.norm(R2, ord=p, axis=0)
        denom = normA[p] * normX + normB
        etas[p] = float(np.max(np.divide(normR, denom,
                                         out=np.zeros_like(normR),
                                         where=denom > 0)))

    comp_denom = _abs_matvec(As, X2) + np.abs(B2)
    omega = float(np.max(np.divide(np.abs(R2), comp_denom,
                                   out=np.zeros_like(comp_denom),
                                   where=comp_denom > 0)))
    return R, etas, omega


def bench(As, B, idx, bs, dtypes=DEFAULT_DTYPES, h5file=None, save=True,
          solvers=DEFAULT_SOLVERS, dtype=None, exclude=None,
          check_blocks=True, fp16_inv_dtype=np.float32):
    """
    Run every requested solver on one system and record the metrics.

    Parameters
    ----------
    bs : int or sequence
        Block partition for the Block Thomas solvers: an int for a uniform
        partition, or a sequence of per-block sizes summing to n for a custom
        one, such as the output of block_sizes_from_matrix(As).
    exclude : dict[str, set[str]], optional
        Maps a solver name to a set of dtype names to skip for that solver, for
        example {"gmres": {"complex64"}}. Other solvers and other precisions
        are unaffected.
    check_blocks : bool
        Verify that bs is a block-tridiagonal partition of As before any Block
        Thomas variant runs. A partition that cuts a real coupling does not
        raise; it discards the coupling and returns a plausible but wrong
        solution. The check costs one pass over the nonzeros.
    fp16_inv_dtype : dtype
        Precision in which BlockThomasExplicitInvFP16 forms its explicit
        inverses before rounding them to float16.

    Returns
    -------
    dict with one entry per (solver, dtype) that ran, plus "idx" and "dtypes".
    """
    if dtype is not None:                      # single-precision legacy call
        dtypes = (dtype,)
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    exclude = exclude or {}

    def _excluded(solver_name, dt):
        return np.dtype(dt).name in exclude.get(solver_name, ())

    n = As.shape[0]
    nb = np.linalg.norm(B)
    # Matrix-only, so computed once per index and reused for every solver and
    # precision. The p=2 case is a sparse SVD and is the expensive one of the
    # three; it still costs one such call per index rather than per solve.
    normA = {p: _matrix_norm(As, p) for p in NORMWISE_ORDS}
    if normA[2] is None:
        print(f"idx={idx}: svds did not converge for ||As||_2; "
              f"eta_2 is skipped for every solver at this index")
    results = {"idx": idx, "dtypes": [np.dtype(d).name for d in dtypes]}
    x_base = None                              # SuperLU solution at dtypes[0]
                                               # against which vs_base is taken

    print(f"idx={idx}  n={n}  B.shape={B.shape}  "
          f"nnz={As.nnz / (n * n):6.2%} of dense  "
          f"baseline=superlu_{_sfx(dtypes[0])}")

    def _finish(key, label, solver, x, t_f, t_s, extra="", saver=None):
        """Record one (solver, dtype) result, report it, and persist it."""
        r, etas, omega = backward_errors(As, x, B, normA)
        res = np.linalg.norm(r) / nb
        mem = solver.factor_nbytes()
        entry = {"factor": t_f, "solve": t_s, "res": res, "mem": mem,
                 "backward_error_normwise_1": etas[1],
                 "backward_error_normwise_2": etas[2],
                 "backward_error_normwise_inf": etas[np.inf],
                 "backward_error_componentwise": omega}
        if x_base is not None and x is not x_base:
            entry["vs_base"] = np.linalg.norm(x - x_base) / np.linalg.norm(x_base)
            extra += f"  vs base {entry['vs_base']:.1e}"
        eta2_str = f"{etas[2]:.1e}" if etas[2] is not None else "n/a"
        extra += (f"  eta1 {etas[1]:.1e}  eta2 {eta2_str}  "
                  f"etaInf {etas[np.inf]:.1e}  omega {omega:.1e}")
        _line(label, t_f, t_s, res, mem, extra)
        results[key] = entry
        if save and h5file is not None and saver is not None:
            saver(x, t_f, t_s, mem, r, etas, omega)
        return entry

    # The blocks depend on As alone, not on the precision, so they are
    # extracted once and reused for every requested precision.
    bt_solvers = [s for s in solvers
                  if s in ("block-thomas", "block-thomas-inv") + FP16_SOLVERS]
    if bt_solvers:
        if check_blocks:
            bad = offband_nnz(As, bs)
            if bad:
                raise ValueError(
                    f"idx={idx}: the requested partition leaves {bad} nonzeros "
                    f"outside the block-tridiagonal band, so the Block Thomas "
                    f"blocks would discard real couplings. Pass a valid "
                    f"partition (see solver_classes.block_sizes_from_matrix) "
                    f"or bench(..., check_blocks=False) to override.")
        D, Lb, Ub = extract_blocks_sparse(As, bs)

    for dt in dtypes:
        sfx = _sfx(dt)

        # ---- SuperLU: the baseline at dtypes[0], an ordinary entry after ----
        if "superlu" in solvers and not _excluded("superlu", dt):
            slu, t_f = _timed(SparseLU, As, dt)
            xs,  t_s = _timed(slu.solve, B)
            if x_base is None:
                x_base = xs
            _finish(f"superlu_{sfx}", f"superlu {sfx}", slu, xs, t_f, t_s,
                    saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_superlu(
                        h5file, dt, idx, slu, x, tf, ts, mem=mem,
                        residual=r, eta1=etas[1], eta2=etas[2],
                        eta_inf=etas[np.inf], omega=omega))

        # ---- UMFPACK: double precision only ------------------------------
        if "umfpack" in solvers and not _excluded("umfpack", dt):
            try:
                umf, t_f = _timed(UMFPACK, As, dt)
                xu,  t_s = _timed(umf.solve, B)
                _finish(f"umfpack_{sfx}", f"umfpack {sfx}", umf, xu, t_f, t_s,
                        saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_umfpack(
                            h5file, dt, idx, umf, x, tf, ts, mem=mem,
                            residual=r, eta1=etas[1], eta2=etas[2],
                            eta_inf=etas[np.inf], omega=omega))
            except (ImportError, TypeError) as e:
                print(f"  umfpack {sfx:12s}: skipped ({e})")
        elif "umfpack" in solvers:
            print(f"  umfpack {sfx:12s}: skipped (excluded)")

        # ---- MUMPS: no explicit factors, so only x and timings are saved --
        if "mumps" in solvers and not _excluded("mumps", dt):
            try:
                mmp, t_f = _timed(MUMPS, As, dt)
                xm,  t_s = _timed(mmp.solve, B)
                _finish(f"mumps_{sfx}", f"mumps {sfx}", mmp, xm, t_f, t_s,
                        extra="  (no L/U exposed)",
                        saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_metadata(
                            h5file, h5_group("mumps"), dt, idx, x, tf, ts, mem=mem,
                            residual=r, eta1=etas[1], eta2=etas[2],
                            eta_inf=etas[np.inf], omega=omega))
            except ImportError as e:
                print(f"  mumps {sfx:14s}: skipped ({e})")
        elif "mumps" in solvers:
            print(f"  mumps {sfx:14s}: skipped (excluded)")

        # ---- GMRES on the CPU, through SciPy ------------------------------
        if "gmres" in solvers and not _excluded("gmres", dt):
            gm, t_f = _timed(GMRES, As, dt)
            xg, t_s = _timed(gm.solve, B)
            _finish(f"gmres_{sfx}", f"gmres (scipy) {sfx}", gm, xg, t_f, t_s,
                    extra=f"  (it~{gm.last_iters})",
                    saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_metadata(
                        h5file, h5_group("gmres"), dt, idx, x, tf, ts,
                        metadata={"iters": gm.last_iters, "info": gm.last_info},
                        mem=mem, residual=r, eta1=etas[1], eta2=etas[2],
                        eta_inf=etas[np.inf], omega=omega))
        elif "gmres" in solvers:
            print(f"  gmres (scipy) {sfx:7s}: skipped (excluded)")

        # ---- GMRES on the GPU, through CuPy -------------------------------
        if "gmres-cupy" in solvers and not _excluded("gmres-cupy", dt):
            if not gpu_available():
                print(f"  gmres (cupy) {sfx:7s}: skipped (no GPU / CuPy not installed)")
            else:
                try:
                    # The construction step is the host-to-device transfer.
                    gmc, t_f = _timed(GMRESCuPy, As, dt)
                    xgc, t_s = _timed(gmc.solve, B)
                    _finish(f"gmres-cupy_{sfx}", f"gmres (cupy) {sfx}", gmc,
                            xgc, t_f, t_s, extra=f"  (it~{gmc.last_iters})",
                            saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_metadata(
                                h5file, "gmres-cupy", dt, idx, x, tf, ts,
                                metadata={"iters": gmc.last_iters,
                                          "info": gmc.last_info},
                                mem=mem, residual=r, eta1=etas[1], eta2=etas[2],
                                eta_inf=etas[np.inf], omega=omega))
                except Exception as e:
                    print(f"  gmres (cupy) {sfx:7s}: FAILED ({type(e).__name__}: {e})")
        elif "gmres-cupy" in solvers:
            print(f"  gmres (cupy) {sfx:7s}: skipped (excluded)")

        # ---- cuDSS: GPU direct solver, no explicit factor values ----------
        if "cudss" in solvers and not _excluded("cudss", dt):
            if not gpu_available():
                print(f"  cudss {sfx:14s}: skipped (no GPU)")
            else:
                try:
                    nrhs = B.shape[1] if B.ndim == 2 else 1
                    cud, t_f = _timed(CuDSS, As, dt, nrhs)
                    xc,  t_s = _timed(cud.solve, B)
                    _finish(f"cudss_{sfx}", f"cudss {sfx}", cud, xc, t_f, t_s,
                            extra="  (no L/U values exposed)",
                            saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_metadata(
                                h5file, h5_group("cudss"), dt, idx, x, tf, ts,
                                metadata=cud.get_metadata(), mem=mem,
                                residual=r, eta1=etas[1], eta2=etas[2],
                                eta_inf=etas[np.inf], omega=omega))
                    cud.free()
                except Exception as e:
                    print(f"  cudss {sfx:14s}: FAILED ({type(e).__name__}: {e})")
        elif "cudss" in solvers:
            print(f"  cudss {sfx:14s}: skipped (excluded)")

        # ---- Block Thomas, implementation 1: LU with substitution ---------
        if "block-thomas" in solvers and not _excluded("block-thomas", dt):
            bt,  t_f = _timed(BlockThomas, D, Lb, Ub, dt)
            xbt, t_s = _timed(bt.solve, B)
            _finish(f"block-thomas_{sfx}", f"block Thomas {sfx}", bt, xbt,
                    t_f, t_s,
                    saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_blockthomas(
                        h5file, dt, idx, bt, x, tf, ts, mem=mem,
                        residual=r, eta1=etas[1], eta2=etas[2],
                        eta_inf=etas[np.inf], omega=omega))
        elif "block-thomas" in solvers:
            print(f"  block Thomas {sfx:7s}: skipped (excluded)")

        # ---- Block Thomas, implementation 2: explicit inverses ------------
        if "block-thomas-inv" in solvers and not _excluded("block-thomas-inv", dt):
            bti,  t_f = _timed(BlockThomasExplicitInv, D, Lb, Ub, dt)
            xbti, t_s = _timed(bti.solve, B)
            _finish(f"block-thomas-inv_{sfx}", f"block Thomas inv {sfx}", bti,
                    xbti, t_f, t_s, extra="  (explicit inverses)",
                    saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_blockthomas_inv(
                        h5file, dt, idx, bti, x, tf, ts, mem=mem,
                        residual=r, eta1=etas[1], eta2=etas[2],
                        eta_inf=etas[np.inf], omega=omega))
        elif "block-thomas-inv" in solvers:
            print(f"  block Thomas inv {sfx:3s}: skipped (excluded)")

    # ---- Block Thomas in half precision --------------------------------------
    # Deliberately outside the precision loop: both variants are
    # precision-fixed and ignore `dtypes`, so running them once per precision
    # would repeat identical work. Results are stored under the label
    # "complex32", which is not a NumPy dtype.
    if "block-thomas-fp16" in solvers:
        try:
            bt16,  t_f = _timed(BlockThomasFP16, D, Lb, Ub)
            xbt16, t_s = _timed(bt16.solve, B)
            _finish("block-thomas-fp16", "block Thomas fp16", bt16, xbt16,
                    t_f, t_s, extra="  (embedded real)",
                    saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_blockthomas(
                        h5file, None, idx, bt16, x, tf, ts, mem=mem,
                        dname=FP16_LABEL, residual=r, eta1=etas[1], eta2=etas[2],
                        eta_inf=etas[np.inf], omega=omega))
        except (FloatingPointError, ZeroDivisionError) as e:
            print(f"  block Thomas fp16   : FAILED ({type(e).__name__}: {e})")

    if "block-thomas-inv-fp16" in solvers:
        try:
            bti16,  t_f = _timed(BlockThomasExplicitInvFP16, D, Lb, Ub,
                                 None, fp16_inv_dtype)
            xbti16, t_s = _timed(bti16.solve, B)
            _finish("block-thomas-inv-fp16", "block Thomas inv fp16", bti16,
                    xbti16, t_f, t_s,
                    extra=f"  (inv in {np.dtype(fp16_inv_dtype).name})",
                    saver=lambda x, tf, ts, mem, r, etas, omega: fio.save_blockthomas_inv(
                        h5file, None, idx, bti16, x, tf, ts, mem=mem,
                        dname=FP16_LABEL, residual=r, eta1=etas[1], eta2=etas[2],
                        eta_inf=etas[np.inf], omega=omega))
        except (FloatingPointError, ZeroDivisionError) as e:
            print(f"  block Thomas inv fp16: FAILED ({type(e).__name__}: {e})")

    print()
    return results
