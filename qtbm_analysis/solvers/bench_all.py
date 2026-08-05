"""
bench_all.py -- extended benchmark driver.  This is the ONLY bench()
implementation -- both the notebook and run_benchmarks.py import it from
here, so there is exactly one place to fix bugs or add solvers.

Import and use:

    from bench_all import bench, DEFAULT_SOLVERS

    metrics = []
    with h5py.File(lu_path, "a") as f:
        for idx in sweep:
            if rhs[idx].shape[-1] == 0:
                continue
            m = bench(M_arr_sp[idx], rhs[idx], idx, bs,
                      dtypes=(np.complex128, np.complex64),
                      h5file=f, save=True)
            metrics.append(m)

Every solver in `solvers` is run once per dtype in `dtypes`.  The FIRST dtype
defines the baseline: SuperLU at dtypes[0] is what every other (solver, dtype)
combination's factor/solve speedup and "vs base" solution error refer to.

Result keys are "<solver>_<suffix>", e.g.:
    superlu_c128, superlu_c64, umfpack_c128, mumps_c64, block_thomas_c64,
    block_thomas_inv_c128, ...

The two fp16 Block Thomas variants are the exception: they are
precision-fixed, run once per index outside the dtype loop, and use the
unsuffixed keys "block_thomas_fp16" / "block_thomas_inv_fp16". They are not
in DEFAULT_SOLVERS -- the fp16 kernels are pure python and far slower than
LAPACK, so ask for them explicitly (bench_all.FP16_SOLVERS) when you want the
accuracy data rather than timings.

The 4th argument `bs` accepts either an int (uniform partition) or a sequence
of per-block sizes (custom partition, e.g. from
solver_classes.block_sizes_from_matrix). The partition is validated against
the matrix before any Block Thomas variant runs.

Skips are graceful and per-(solver, dtype): UMFPACK has no single-precision
build so umfpack_c64 prints a skip line; GPU solvers (gmres_cupy, cudss) are
dropped on CPU-only machines; a missing package skips that solver entirely.

`exclude` lets you drop specific (solver, dtype) combinations on purpose,
e.g. to skip GMRES at single precision across a whole batch run:

    bench(..., exclude={"gmres": {"complex64"}, "gmres_cupy": {"complex64"}})

Each solver's factor_nbytes() is passed straight through to factor_io's
savers as `mem`, which store it as a "factor_nbytes" HDF5 attribute -- no
monkeypatching of factor_io required.

Backwards compatibility: bench(..., dtype=np.complex128) still works and is
treated as dtypes=(np.complex128,).
"""

import time
import numpy as np

from solver_classes import (
    SparseLU, GMRES, extract_blocks_sparse, offband_nnz,
    BlockThomas, BlockThomasExplicitInv,
    BlockThomasFP16, BlockThomasExplicitInvFP16,
    UMFPACK, MUMPS, GMRESCuPy, CuDSS, gpu_available,
)
import factor_io as fio

DEFAULT_SOLVERS = ("superlu", "umfpack", "mumps", "gmres",
                   "gmres_cupy", "cudss", "block_thomas", "block_thomas_inv")

# fp16 Block Thomas is precision-fixed: it ignores the dtype loop and runs
# exactly once per index, stored under the label "complex32". Not in
# DEFAULT_SOLVERS -- the pure-python fp16 kernels are orders of magnitude
# slower than LAPACK, so opt in explicitly when you want the accuracy data.
FP16_SOLVERS = ("block_thomas_fp16", "block_thomas_inv_fp16")

DEFAULT_DTYPES = (np.complex128, np.complex64)

_SUFFIX = {"complex128": "c128", "complex64": "c64",
           "float64": "f64", "float32": "f32"}
FP16_LABEL = "complex32"       # not a numpy dtype -- a storage label only


def _sfx(dt):
    name = np.dtype(dt).name
    return _SUFFIX.get(name, name)


def _timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t0


def _line(label, t_f, t_s, res, mem, extra=""):
    print(f"  {label:20s}: factor {t_f*1e3:8.2f} ms  solve {t_s*1e3:8.3f} ms"
          f"  res {res:.1e}  mem {mem/1e6:7.1f} MB{extra}")


def bench(As, B, idx, bs, dtypes=DEFAULT_DTYPES, h5file=None, save=True,
          solvers=DEFAULT_SOLVERS, dtype=None, exclude=None,
          check_blocks=True, fp16_inv_dtype=np.float32):
    """
    bs      : the block partition for the Block Thomas solvers -- an int for a
              uniform partition (historical behaviour) or a sequence of
              per-block sizes summing to n for a custom one, e.g. the output
              of solver_classes.block_sizes_from_matrix(As). Everything else
              in the signature is unchanged.

    exclude : optional dict[str, set[str]] mapping a base solver name
              ("gmres", "gmres_cupy", "umfpack", ...) to a set of dtype
              *names* ("complex64", "complex128") to skip for that solver.
              e.g. {"gmres": {"complex64"}} skips GMRES only at single
              precision; other solvers and other dtypes are unaffected.

    check_blocks : verify that `bs` really is a block-tridiagonal partition of
              As before running any Block Thomas variant. A partition that
              cuts through real couplings does not raise -- it silently drops
              them and returns a plausible but wrong x -- so this is on by
              default and costs one pass over the nonzeros.

    fp16_inv_dtype : precision in which BlockThomasExplicitInvFP16 forms its
              explicit inverses before rounding them to fp16 (default fp32).
    """
    if dtype is not None:                      # old single-dtype signature
        dtypes = (dtype,)
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    exclude = exclude or {}

    def _excluded(solver_name, dt):
        return np.dtype(dt).name in exclude.get(solver_name, ())

    n = As.shape[0]
    nb = np.linalg.norm(B)
    results = {"idx": idx, "dtypes": [np.dtype(d).name for d in dtypes]}
    x_base = None                              # SuperLU solution at dtypes[0]

    print(f"idx={idx}  n={n}  B.shape={B.shape}  "
          f"nnz={As.nnz / (n * n):6.2%} of dense  "
          f"baseline=superlu_{_sfx(dtypes[0])}")

    def _finish(key, label, solver, x, t_f, t_s, extra="", saver=None):
        res = np.linalg.norm(As @ x - B) / nb
        mem = solver.factor_nbytes()
        entry = {"factor": t_f, "solve": t_s, "res": res, "mem": mem}
        if x_base is not None and x is not x_base:
            entry["vs_base"] = np.linalg.norm(x - x_base) / np.linalg.norm(x_base)
            extra += f"  vs base {entry['vs_base']:.1e}"
        _line(label, t_f, t_s, res, mem, extra)
        results[key] = entry
        if save and h5file is not None and saver is not None:
            saver(x, t_f, t_s, mem)
        return entry

    # blocks depend only on As, not on dtype -> extract once, reuse per dtype
    bt_solvers = [s for s in solvers
                  if s in ("block_thomas", "block_thomas_inv") + FP16_SOLVERS]
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

        # ---- SuperLU (baseline at dtypes[0], regular entry otherwise) ----
        if "superlu" in solvers and not _excluded("superlu", dt):
            slu, t_f = _timed(SparseLU, As, dt)
            xs,  t_s = _timed(slu.solve, B)
            if x_base is None:
                x_base = xs
            _finish(f"superlu_{sfx}", f"superlu {sfx}", slu, xs, t_f, t_s,
                    saver=lambda x, tf, ts, mem: fio.save_superlu(
                        h5file, dt, idx, slu, x, tf, ts, mem=mem))

        # ---- UMFPACK (double precision only) -----------------------------
        if "umfpack" in solvers and not _excluded("umfpack", dt):
            try:
                umf, t_f = _timed(UMFPACK, As, dt)
                xu,  t_s = _timed(umf.solve, B)
                _finish(f"umfpack_{sfx}", f"umfpack {sfx}", umf, xu, t_f, t_s,
                        saver=lambda x, tf, ts, mem: fio.save_umfpack(
                            h5file, dt, idx, umf, x, tf, ts, mem=mem))
            except (ImportError, TypeError) as e:
                print(f"  umfpack {sfx:12s}: skipped ({e})")
        elif "umfpack" in solvers:
            print(f"  umfpack {sfx:12s}: skipped (excluded)")

        # ---- MUMPS (no explicit factors -- x + timings saved) ------------
        if "mumps" in solvers and not _excluded("mumps", dt):
            try:
                mmp, t_f = _timed(MUMPS, As, dt)
                xm,  t_s = _timed(mmp.solve, B)
                _finish(f"mumps_{sfx}", f"mumps {sfx}", mmp, xm, t_f, t_s,
                        extra="  (no L/U exposed)",
                        saver=lambda x, tf, ts, mem: fio.save_metadata(
                            h5file, "mumps", dt, idx, x, tf, ts, mem=mem))
            except ImportError as e:
                print(f"  mumps {sfx:14s}: skipped ({e})")
        elif "mumps" in solvers:
            print(f"  mumps {sfx:14s}: skipped (excluded)")

        # ---- GMRES (SciPy, CPU) -------------------------------------------
        if "gmres" in solvers and not _excluded("gmres", dt):
            gm, t_f = _timed(GMRES, As, dt)
            xg, t_s = _timed(gm.solve, B)
            _finish(f"gmres_{sfx}", f"gmres (scipy) {sfx}", gm, xg, t_f, t_s,
                    extra=f"  (it~{gm.last_iters})",
                    saver=lambda x, tf, ts, mem: fio.save_metadata(
                        h5file, "gmres_scipy", dt, idx, x, tf, ts,
                        metadata={"iters": gm.last_iters, "info": gm.last_info},
                        mem=mem))
        elif "gmres" in solvers:
            print(f"  gmres (scipy) {sfx:7s}: skipped (excluded)")

        # ---- GMRES (CuPy, GPU only) ----------------------------------------
        if "gmres_cupy" in solvers and not _excluded("gmres_cupy", dt):
            if not gpu_available():
                print(f"  gmres (cupy) {sfx:7s}: skipped (no GPU / CuPy not installed)")
            else:
                try:
                    gmc, t_f = _timed(GMRESCuPy, As, dt)  # factor == H->D transfer
                    xgc, t_s = _timed(gmc.solve, B)
                    _finish(f"gmres_cupy_{sfx}", f"gmres (cupy) {sfx}", gmc,
                            xgc, t_f, t_s, extra=f"  (it~{gmc.last_iters})",
                            saver=lambda x, tf, ts, mem: fio.save_metadata(
                                h5file, "gmres_cupy", dt, idx, x, tf, ts,
                                metadata={"iters": gmc.last_iters,
                                          "info": gmc.last_info},
                                mem=mem))
                except Exception as e:
                    print(f"  gmres (cupy) {sfx:7s}: FAILED ({type(e).__name__}: {e})")
        elif "gmres_cupy" in solvers:
            print(f"  gmres (cupy) {sfx:7s}: skipped (excluded)")

        # ---- cuDSS (GPU only, no explicit factor values) -------------------
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
                            saver=lambda x, tf, ts, mem: fio.save_metadata(
                                h5file, "cudss", dt, idx, x, tf, ts,
                                metadata=cud.get_metadata(), mem=mem))
                    cud.free()
                except Exception as e:
                    print(f"  cudss {sfx:14s}: FAILED ({type(e).__name__}: {e})")
        elif "cudss" in solvers:
            print(f"  cudss {sfx:14s}: skipped (excluded)")

        # ---- Block Thomas, Implementation 1 (LU + substitution) -------------
        if "block_thomas" in solvers and not _excluded("block_thomas", dt):
            bt,  t_f = _timed(BlockThomas, D, Lb, Ub, dt)
            xbt, t_s = _timed(bt.solve, B)
            _finish(f"block_thomas_{sfx}", f"block Thomas {sfx}", bt, xbt,
                    t_f, t_s,
                    saver=lambda x, tf, ts, mem: fio.save_blockthomas(
                        h5file, dt, idx, bt, x, tf, ts, mem=mem))
        elif "block_thomas" in solvers:
            print(f"  block Thomas {sfx:7s}: skipped (excluded)")

        # ---- Block Thomas, Implementation 2 (explicit inverses) -------------
        if "block_thomas_inv" in solvers and not _excluded("block_thomas_inv", dt):
            bti,  t_f = _timed(BlockThomasExplicitInv, D, Lb, Ub, dt)
            xbti, t_s = _timed(bti.solve, B)
            _finish(f"block_thomas_inv_{sfx}", f"block Thomas inv {sfx}", bti,
                    xbti, t_f, t_s, extra="  (explicit inverses)",
                    saver=lambda x, tf, ts, mem: fio.save_blockthomas_inv(
                        h5file, dt, idx, bti, x, tf, ts, mem=mem))
        elif "block_thomas_inv" in solvers:
            print(f"  block Thomas inv {sfx:3s}: skipped (excluded)")

    # ---- fp16 Block Thomas ---------------------------------------------------
    # Outside the dtype loop on purpose: both fp16 variants are precision-fixed
    # and ignore `dtypes` entirely, so running them once per dtype would just
    # repeat identical work. Results are stored under the label "complex32".
    if "block_thomas_fp16" in solvers:
        try:
            bt16,  t_f = _timed(BlockThomasFP16, D, Lb, Ub)
            xbt16, t_s = _timed(bt16.solve, B)
            _finish("block_thomas_fp16", "block Thomas fp16", bt16, xbt16,
                    t_f, t_s, extra="  (embedded real)",
                    saver=lambda x, tf, ts, mem: fio.save_blockthomas(
                        h5file, None, idx, bt16, x, tf, ts, mem=mem,
                        dname=FP16_LABEL))
        except (FloatingPointError, ZeroDivisionError) as e:
            print(f"  block Thomas fp16   : FAILED ({type(e).__name__}: {e})")

    if "block_thomas_inv_fp16" in solvers:
        try:
            bti16,  t_f = _timed(BlockThomasExplicitInvFP16, D, Lb, Ub,
                                 None, fp16_inv_dtype)
            xbti16, t_s = _timed(bti16.solve, B)
            _finish("block_thomas_inv_fp16", "block Thomas inv fp16", bti16,
                    xbti16, t_f, t_s,
                    extra=f"  (inv in {np.dtype(fp16_inv_dtype).name})",
                    saver=lambda x, tf, ts, mem: fio.save_blockthomas_inv(
                        h5file, None, idx, bti16, x, tf, ts, mem=mem,
                        dname=FP16_LABEL))
        except (FloatingPointError, ZeroDivisionError) as e:
            print(f"  block Thomas inv fp16: FAILED ({type(e).__name__}: {e})")

    print()
    return results
