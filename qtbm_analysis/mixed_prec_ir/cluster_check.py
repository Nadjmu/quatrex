"""Run on the cluster: is double-double actually faster than clongdouble there?

    python cluster_check.py /scratch/yimili/matrices2/hdf5/graphene.h5 209
"""
import sys, time
sys.path.insert(0, ".")
sys.path.append("../solvers")
import numpy as np, mpir

h5path, idx = sys.argv[1], int(sys.argv[2])
A, b = mpir.load_system(h5path, idx)
A = A.tocsc().astype(np.complex128)
b = np.asarray(b, dtype=np.complex128)
print(f"{h5path} E_{idx}: n={A.shape[0]} nnz={A.nnz} rhs={b.shape[1]}")
print(f"np.longdouble eps = {float(np.finfo(np.longdouble).eps):.2e} "
      f"({np.dtype(np.longdouble).itemsize*8}-bit)")

t = time.perf_counter(); mpir._ExtendedReference(A).solve(b)
t_dd = time.perf_counter() - t
print(f"  double-double (new) : {t_dd:7.2f}s")

EXT = np.clongdouble
class Old(mpir._ExtendedReference):
    def _mv(self, x):
        ip = self._csr.indptr
        pr = self._csr.data.astype(EXT) * x.astype(EXT)[self._csr.indices]
        out = np.zeros(self._csr.shape[0], dtype=EXT)
        ne = np.flatnonzero(np.diff(ip) > 0)
        out[ne] = np.add.reduceat(pr, ip[ne])
        return out
    def _solve_one(self, bb):
        bb = np.asarray(bb, dtype=np.complex128)
        x = self._lu.solve(bb)
        tol = float(np.finfo(np.longdouble).eps) * 10
        for _ in range(self._max_steps):
            r = bb.astype(EXT) - self._mv(x)
            d = self._lu.solve(np.asarray(r, dtype=np.complex128))
            x = x.astype(EXT) + d.astype(EXT)
            self.steps += 1
            nx = float(np.max(np.abs(x)))
            if nx and float(np.max(np.abs(d))) <= tol * nx:
                break
        return np.asarray(x, dtype=np.complex128)

t = time.perf_counter(); Old(A).solve(b)
t_old = time.perf_counter() - t
print(f"  clongdouble (old)   : {t_old:7.2f}s")
print(f"\n  double-double is {t_old/t_dd:.1f}x the speed of clongdouble HERE")
print("  (>1 means the change is a win on this machine, <1 means it is a loss)")
