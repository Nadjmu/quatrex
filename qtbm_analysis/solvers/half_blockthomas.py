"""
blockthomas_fp16.py -- BlockThomasFP16: Higham Alg. 13.3 Implementation 1
(LU + substitution, no explicit inverses) with ALL block arithmetic in fp16.

Complex blocks are handled via the exact real embedding
    z = a+bi  ->  [[a,-b],[b,a]]   (2m x 2m real per m x m complex block)
so every operation is real fp16 -- no complex32 dtype dependency.

fp16 backend: numpy float16. numpy computes each op in fp32 internally and
rounds the result back to fp16 (chop-style per-op rounding, same accumulate
semantics as tensor cores). A torch/GPU backend is a mechanical swap later.

Contract matches solver_classes:
    __init__(D, L, U)   full factorization (this is what you time)
    solve(b)            complex b (n,) or (n, nrhs) -> complex, same shape
    get_LUP()           (Lb, Ub, Dmod_lu, Dmod_piv) -- embedded real fp16,
                        block size 2*bs; plus scale attr self.s
    factor_nbytes()

Naming follows solver_classes.BlockThomas: L = sub-diagonal blocks,
U = super-diagonal blocks, D = diagonal blocks.

Intended home: paste the class (+ the three _fp16 helpers + embed utils)
into solver_classes.py next to BlockThomas / BlockThomasExplicitInv.
H5 note: factors are real float16 with block size 2*bs; save under a
dtype name like "complex32" (np.dtype("complex32") doesn't exist, so
save_blockthomas needs its `dname = np.dtype(dtype).name` line bypassed --
pass dname="complex32" directly).
"""

import numpy as np

H = np.float16


# ---------- complex <-> real embedding (exact) ----------

def embed_block(Z):
    a, b = np.ascontiguousarray(Z.real), np.ascontiguousarray(Z.imag)
    return np.block([[a, -b], [b, a]])

def embed_vec(z):
    return np.concatenate([z.real, z.imag])

def unembed_vec(v):
    m = v.shape[0] // 2
    return v[:m] + 1j * v[m:]


# ---------- fp16 dense LU (partial pivoting) + multi-RHS solve ----------

def lu_fp16(Dm):
    """Packed LU of (k,k) fp16 array, fp16 rounding per op. Returns LU, piv."""
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
    """Solve (LU) X = Rhs in fp16. Rhs: (k,) or (k, r)."""
    X = Rhs.astype(H)[piv].copy()
    one_d = X.ndim == 1
    if one_d:
        X = X[:, None]
    k = LU.shape[0]
    for j in range(k):                       # unit-L forward
        X[j + 1:] = (X[j + 1:] - np.outer(LU[j + 1:, j], X[j])).astype(H)
    for j in range(k - 1, -1, -1):           # U backward
        X[j] = (X[j] / LU[j, j]).astype(H)
        X[:j] = (X[:j] - np.outer(LU[:j, j], X[j])).astype(H)
    return X[:, 0] if one_d else X


# ---------- solver class ----------

class BlockThomasFP16:
    """Block Thomas in fp16 (Implementation 1: LU + substitution).

    D, L, U: lists of complex (bs, bs) blocks -- same inputs as BlockThomas
    (from extract_blocks_sparse). All internal arithmetic on embedded real
    fp16 blocks of size (2*bs, 2*bs).
    """

    def __init__(self, D, L, U, dtype=None):   # dtype ignored (always fp16)
        self.N = len(D)
        self.bs = D[0].shape[0]
        self.dtype = H

        # global power-of-2 scaling (exact) so entries sit inside fp16 range
        amax = max(float(np.abs(embed_block(d)).max()) for d in D)
        self.s = 2.0 ** np.floor(np.log2(1024.0 / amax))
        to16 = lambda X: (embed_block(X) * self.s).astype(H)

        self.L = [to16(l) for l in L]           # sub-diagonal
        self.U = [to16(u) for u in U]           # super-diagonal
        self.lu_piv = [None] * self.N
        Dm = to16(D[0])
        self.lu_piv[0] = lu_fp16(Dm)
        for k in range(1, self.N):
            # D_mod[k] = D[k] - L[k-1] @ D_mod[k-1]^{-1} @ U[k-1], via lu_solve
            W = lu_solve_fp16(*self.lu_piv[k - 1], self.U[k - 1])
            Dm = (to16(D[k]) - (self.L[k - 1] @ W).astype(H)).astype(H)
            self.lu_piv[k] = lu_fp16(Dm)

    # -- solve --------------------------------------------------------------
    def _solve_one(self, b):
        # rescale rhs to O(1): fp16 underflows below ~6e-5. Power of 2: exact.
        rmax = float(np.abs(b).max())
        if rmax == 0:
            return np.zeros_like(b)
        rs = 2.0 ** np.floor(np.log2(1.0 / rmax))
        bs = self.bs
        bb = [embed_vec(b[k * bs:(k + 1) * bs] * rs).astype(H)
              for k in range(self.N)]
        for k in range(1, self.N):           # forward sweep
            w = lu_solve_fp16(*self.lu_piv[k - 1], bb[k - 1])
            bb[k] = (bb[k] - (self.L[k - 1] @ w).astype(H)).astype(H)
        x = [None] * self.N
        x[-1] = lu_solve_fp16(*self.lu_piv[-1], bb[-1])
        for k in range(self.N - 2, -1, -1):  # block back substitution
            x[k] = lu_solve_fp16(*self.lu_piv[k],
                                 (bb[k] - (self.U[k] @ x[k + 1]).astype(H)).astype(H))
        out = np.concatenate([unembed_vec(xk.astype(np.float64)) for xk in x])
        # factors approximate (s*A)^{-1} = A^{-1}/s; rhs was scaled by rs
        return out * (self.s / rs)

    def solve(self, b):
        b = np.asarray(b)
        if b.ndim == 1:
            return self._solve_one(b)
        return np.column_stack([self._solve_one(np.ascontiguousarray(b[:, j]))
                                for j in range(b.shape[1])])

    # -- factor access (solver_classes contract) ----------------------------
    def get_LUP(self):
        """Stacked embedded-real fp16 factors, block size 2*bs.
        Reconstruction needs self.s: factors are of s*A_embedded."""
        b2 = 2 * self.bs
        Lb = np.stack(self.L) if self.L else np.empty((0, b2, b2), H)
        Ub = np.stack(self.U) if self.U else np.empty((0, b2, b2), H)
        Dlu = np.stack([lu for lu, piv in self.lu_piv])
        Dpiv = np.stack([piv for lu, piv in self.lu_piv])
        return Lb, Ub, Dlu, Dpiv

    def factor_nbytes(self):
        Lb, Ub, Dlu, Dpiv = self.get_LUP()
        return int(Lb.nbytes + Ub.nbytes + Dlu.nbytes + Dpiv.nbytes)


# ---------- self-test ----------

if __name__ == "__main__":
    import scipy.sparse as sp
    import scipy.linalg as sla

    # synthetic complex block-tridiagonal system, assembled sparse and pulled
    # apart with the same block extraction the bench uses
    def extract_blocks_sparse(As, bs):
        n = As.shape[0]; N = n // bs
        Ac = As.tocsr()
        D = [np.asarray(Ac[k*bs:(k+1)*bs, k*bs:(k+1)*bs].todense()) for k in range(N)]
        L = [np.asarray(Ac[(k+1)*bs:(k+2)*bs, k*bs:(k+1)*bs].todense()) for k in range(N-1)]
        U = [np.asarray(Ac[k*bs:(k+1)*bs, (k+1)*bs:(k+2)*bs].todense()) for k in range(N-1)]
        return D, L, U

    rng = np.random.default_rng(0)
    Nb, bs = 30, 8
    rc = lambda: rng.standard_normal((bs, bs)) + 1j * rng.standard_normal((bs, bs))
    Dl = [rc() + 3.0 * bs * np.eye(bs) for _ in range(Nb)]
    Ll = [rc() for _ in range(Nb - 1)]
    Ul = [rc() for _ in range(Nb - 1)]
    As = sp.bmat([[Dl[i] if i == j else
                   Ul[i] if j == i + 1 else
                   Ll[j] if i == j + 1 else None
                   for j in range(Nb)] for i in range(Nb)], format="csc")

    D, L, U = extract_blocks_sparse(As, bs)
    n = Nb * bs
    x_true = rng.standard_normal(n) + 1j * rng.standard_normal(n)
    b = As @ x_true

    bt = BlockThomasFP16(D, L, U)
    x = bt.solve(b)
    relres = np.linalg.norm(As @ x - b) / np.linalg.norm(b)
    ferr = np.linalg.norm(x - x_true, np.inf) / np.linalg.norm(x_true, np.inf)
    print(f"fp16 block Thomas:  relres = {relres:.3e}   fwd err = {ferr:.3e}")
    print(f"(expect ~1e-3, fp16 unit roundoff is 4.9e-4)")

    # reference: fp64 block Thomas on the same blocks (Implementation 1)
    lu0 = sla.lu_factor(D[0]); lus = [lu0]; Dm = D[0]
    for k in range(1, Nb):
        Dm = D[k] - L[k-1] @ sla.lu_solve(lus[-1], U[k-1]); lus.append(sla.lu_factor(Dm))
    bb = [b[k*bs:(k+1)*bs].copy() for k in range(Nb)]
    for k in range(1, Nb):
        bb[k] = bb[k] - L[k-1] @ sla.lu_solve(lus[k-1], bb[k-1])
    xr = [None]*Nb; xr[-1] = sla.lu_solve(lus[-1], bb[-1])
    for k in range(Nb-2, -1, -1):
        xr[k] = sla.lu_solve(lus[k], bb[k] - U[k] @ xr[k+1])
    x64 = np.concatenate(xr)
    print(f"fp64 reference:     relres = {np.linalg.norm(As @ x64 - b)/np.linalg.norm(b):.3e}")
    print(f"fp16 vs fp64 solution: {np.linalg.norm(x - x64, np.inf)/np.linalg.norm(x64, np.inf):.3e}")

    # multi-RHS path
    Bm = np.column_stack([b, 2 * b])
    Xm = bt.solve(Bm)
    print(f"multi-RHS check: {np.linalg.norm(As @ Xm[:,1] - 2*b)/np.linalg.norm(2*b):.3e}")