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
    fn()
    s = time.perf_counter()
    for _ in range(k):
        out = fn()
    return (time.perf_counter() - s) / k, out


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
print()

print("[1] the factorization the reference refines from")
s = time.perf_counter(); lu = spla.splu(A); t_su = time.perf_counter() - s
print(f"    SuperLU      : {t_su:8.2f}s   fill = {(lu.L.nnz+lu.U.nnz)/A.nnz:6.1f}x A.nnz"
      f"   (L+U = {lu.L.nnz+lu.U.nnz:,})")
s = time.perf_counter()
bt = mpir.SOLVER_BUILDERS["block-thomas"](
    A, np.complex128, bs, np.zeros((n, 1), dtype=np.complex128), np.float32)
t_bt = time.perf_counter() - s
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

print("[3] the whole reference, split into its actual parts")
print("    (each twice, so run-to-run variance is visible rather than assumed)")
for label, blocks in (("SuperLU     ", [n]), ("Block Thomas", bs)):
    for rep in range(2):
        s0 = time.perf_counter()
        name, fac = mpir._reference_factorization(A, blocks)
        t_fac = time.perf_counter() - s0

        s0 = time.perf_counter()
        r = mpir._ExtendedReference(A, blocks)
        t_build = time.perf_counter() - s0

        s0 = time.perf_counter(); x = r.solve(b); t_solve = time.perf_counter() - s0
        # how much of the solve is the extended matvec vs the triangular solves
        nmv = r.steps + nrhs                 # one residual per step, plus the first
        print(f"    {label} #{rep+1}: factorize {t_fac:7.2f}s | "
              f"precompute {t_build - t_fac:7.2f}s | solve {t_solve:7.2f}s "
              f"= {t_build + t_solve:7.2f}s   [{name}, steps/col={r.steps//max(nrhs,1)}]")
        del r, fac
