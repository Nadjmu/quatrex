"""Where does the reference solution actually spend its time on this machine?

    python cluster_check.py /scratch/yimili/matrices2/hdf5/graphene.h5 209

Times each phase separately, and SuperLU against Block Thomas as the
factorization the reference refines from. An earlier version of this script
timed the two arithmetics end to end, which shared their setup cost and so
mostly measured that instead.
"""
import sys, time
sys.path.insert(0, ".")
sys.path.append("../solvers")
import numpy as np
import scipy.sparse.linalg as spla
import mpir


def t(fn, k=1):
    """Median of k timed calls after a warm-up, so one contended call cannot
    set the number. Everything here was originally single-sample, which is
    how a 0.37s factorization also measured 17.15s in the same session."""
    fn()
    ts = []
    for _ in range(k):
        s = time.perf_counter(); out = fn(); ts.append(time.perf_counter() - s)
    return float(np.median(ts)), out


h5path, idx = sys.argv[1], int(sys.argv[2])
A, b = mpir.load_system(h5path, idx)
A = A.tocsc().astype(np.complex128)
b = np.asarray(b, dtype=np.complex128)
n, nrhs = A.shape[0], b.shape[1]
counts = np.diff(A.tocsr().indptr)
bs = mpir.block_sizes_from_matrix(A)
print(f"{h5path} E_{idx}")
print(f"  n={n}  nnz={A.nnz}  rhs={nrhs}")
print(f"  row lengths : min={counts.min()} max={counts.max()} "
      f"mean={counts.mean():.0f}  distinct={len(np.unique(counts))}")
print(f"  blocks      : {len(bs)}  sizes {min(bs)}..{max(bs)}")
print(f"  longdouble  : eps={float(np.finfo(np.longdouble).eps):.2e}")
import os
thr = {k: os.environ.get(k, "unset") for k in
       ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}
print(f"  threads     : " + "  ".join(f"{k.split('_')[0]}={v}" for k, v in thr.items())
      + f"   cores={os.cpu_count()}")
if all(v == "unset" for v in thr.values()):
    print("  WARNING: no thread limit set. The factorizations below are BLAS-bound")
    print("           and on a shared node their timings swing by 20-50x. Re-run as")
    print("           OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 python cluster_check.py ...")
    print("           on a COMPUTE node before believing any factorization number.")
print()

print("[1] the factorization the reference refines from")
t_su, lu = t(lambda: spla.splu(A), 3)
print(f"    SuperLU      : {t_su:8.2f}s   fill = {(lu.L.nnz+lu.U.nnz)/A.nnz:6.1f}x A.nnz"
      f"   (L+U = {lu.L.nnz+lu.U.nnz:,})")
t_bt, bt = t(lambda: mpir.SOLVER_BUILDERS["block-thomas"](
    A, np.complex128, bs, np.zeros((n, 1), dtype=np.complex128), np.float32), 3)
print(f"    Block Thomas : {t_bt:8.2f}s"
      f"   -> {t_su/max(t_bt,1e-9):.1f}x cheaper than SuperLU")
x0 = np.ascontiguousarray(b[:, 0])
d1, _ = t(lambda: lu.solve(x0), 3)
d2, _ = t(lambda: bt.solve(x0), 3)
print(f"    one solve    : SuperLU {d1*1000:.1f} ms   Block Thomas {d2*1000:.1f} ms")
print()

print("[2] one extended-precision matvec (per refinement step, per rhs)")
ref = mpir._ExtendedReference(A, bs)
print(f"    row-length buckets in the double-double reduction: {len(ref._blocks)}")
dd, _ = t(lambda: ref._matvec_dd(x0, None), 2)
EXT = np.clongdouble
csr = ref._csr
def mv_old():
    ip = csr.indptr
    pr = csr.data.astype(EXT) * x0.astype(EXT)[csr.indices]
    out = np.zeros(csr.shape[0], dtype=EXT)
    ne = np.flatnonzero(np.diff(ip) > 0)
    out[ne] = np.add.reduceat(pr, ip[ne])
    return out
od, _ = t(mv_old, 2)
pl, _ = t(lambda: csr @ x0, 3)
print(f"    double-double (new) : {dd*1000:9.1f} ms")
print(f"    clongdouble   (old) : {od*1000:9.1f} ms")
print(f"    plain complex128    : {pl*1000:9.1f} ms   (the floor)")
print(f"    -> double-double is {od/dd:.1f}x the speed of clongdouble, "
      f"and {dd/pl:.0f}x a plain matvec")
print()

print("[3] the whole reference, split into its actual parts (median of 3)")
real_fac = mpir._reference_factorization
for label, blocks in (("SuperLU     ", [n]), ("Block Thomas", bs)):
    t_fac, made = t(lambda: real_fac(A, blocks), 3)
    # Hold the factorization fixed so what is timed next is only the
    # double-double precompute, not a second factorization.
    mpir._reference_factorization = lambda A_, b_, _m=made: _m
    try:
        t_pre, r = t(lambda: mpir._ExtendedReference(A, blocks), 3)
    finally:
        mpir._reference_factorization = real_fac
    t_solve, _ = t(lambda: r.solve(b), 3)
    print(f"    {label}: factorize {t_fac:7.2f}s | precompute {t_pre:7.2f}s | "
          f"solve {t_solve:7.2f}s = {t_fac + t_pre + t_solve:7.2f}s"
          f"   [{made[0]}, steps/col={r.steps//max(nrhs,1)}]")
