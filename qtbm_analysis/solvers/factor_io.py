"""
HDF5 persistence of solver factors, solutions and metadata.

Layout
------
Solver output is appended into the material's own HDF5 file, the same file that
holds E_<idx>/M and E_<idx>/rhs; there is no separate factor file. Each
solver's group is a direct sibling of M, Sigma, rhs and spectrum inside the
energy index's own group, so a matrix and every factorization of it are stored
together:

    E_<idx>/superlu/<dtype>/          L, U as sparse CSC, perm_r, perm_c, x,
                                      time_fact, time_solve
    E_<idx>/umfpack/<dtype>/          the same, plus R and do_recip
    E_<idx>/blockthomas/<dtype>/      L, U as dense blocks, Dmod_lu, Dmod_piv,
                                      block_sizes, x, times
    E_<idx>/blockthomas_inv/<dtype>/  L, U, Dmod_inv, Dmod, block_sizes, x,
                                      times
    E_<idx>/mumps/<dtype>/            x and times only; no factors are exposed
    E_<idx>/cudss/<dtype>/            permutations, lu_nnz, x, times
    E_<idx>/gmres*/<dtype>/           x, times, iters, info; iterative, so no
                                      factors

Attributes
----------
Both Block Thomas groups carry: implementation (1 for LU with substitution, 2
for explicit inverses), block_sizes always and block_size for uniform
partitions only, and uniform_blocks. The half-precision variants additionally
carry scale_s and embedded_real, recording that the stored factors are of
s * embed(A) at block size 2m rather than of A, and inv_dtype.

Every group carries factor_nbytes whenever a saver is called with mem set.

The half-precision results are stored under the dtype label "complex32", which
is not a NumPy dtype; the savers must be called with dname="complex32"
explicitly, since np.dtype("complex32") does not exist.

Block partitions
----------------
A uniform partition stores L, U and the Dmod arrays as stacked (N, bs, bs)
arrays. A custom partition has ragged blocks, for which HDF5 has no native
shape; those are flattened into one contiguous 1-D buffer plus an int64 (N, 2)
table of per-block shapes, and the dataset is tagged ragged=True. Read either
layout back with load_blocks(g, name) rather than indexing the dataset
directly.

Overwrite semantics
-------------------
Every save_* call deletes and recreates its own E_<idx>/<solver>/<dtype> group
before writing, so re-running a combination overwrites cleanly and leaves no
stale datasets or attributes. Nothing outside that group is touched, so M, rhs,
Sigma and spectrum are never modified.

The root argument of every function here must be the material's own HDF5 file,
opened in mode "a" or "r+": the same path from which M and rhs were read.

Reconstruction conventions
--------------------------
Also recorded as a group attribute:

    superlu   Pr A Pc == L U                  Pr from argsort(perm_r)
    umfpack   Pr diag(1/R) A Pc == L U        Pr from argsort(perm_r)
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _fresh_group(root, group_path):
    """Delete group_path if present, then recreate it empty."""
    if group_path in root:
        del root[group_path]
    return root.require_group(group_path)


def _solver_group_path(idx, solver_name, dname):
    """
    Group path for one (solver, dtype, energy index) result:

        E_<idx>/<solver_name>/<dtype>

    The solver groups are direct siblings of E_<idx>/M, E_<idx>/rhs and
    E_<idx>/Sigma, with no intermediate level, so solver output is stored
    alongside the matrix that produced it.
    """
    return f"E_{int(idx)}/{solver_name}/{dname}"


def _save_sparse_factor(g, name, mat):
    """Write a sparse matrix as a CSC triplet under g/name, overwriting."""
    sg = g.require_group(name)
    for k in list(sg.keys()):
        del sg[k]
    mat = mat.tocsc()
    sg.create_dataset("data",    data=mat.data,    compression="gzip")
    sg.create_dataset("indices", data=mat.indices, compression="gzip")
    sg.create_dataset("indptr",  data=mat.indptr,  compression="gzip")
    sg.attrs["shape"] = mat.shape


def load_sparse_factor(g):
    """Inverse of _save_sparse_factor: a data/indices/indptr group to CSC."""
    shape = tuple(g.attrs["shape"]) if "shape" in g.attrs else None
    return sp.csc_matrix((g["data"][:], g["indices"][:], g["indptr"][:]),
                         shape=shape)


def _save_dense(g, name, arr, compress=True):
    if name in g:
        del g[name]
    if compress and np.asarray(arr).ndim > 0:
        g.create_dataset(name, data=arr, compression="gzip")
    else:
        g.create_dataset(name, data=arr)


# ---------------------------------------------------------------------------
# block factors: stacked when the partition is uniform, ragged when it is not
# ---------------------------------------------------------------------------
def _save_blocks(g, name, blocks):
    """
    Write one Block Thomas factor part, in either block layout.

    A uniform partition arrives as a stacked (N, bs, bs) array and is written
    unchanged. A custom partition arrives as a list of ragged per-block arrays,
    for which HDF5 has no native shape; those are flattened into one contiguous
    1-D buffer plus an int64 (N, 2) table of per-block shapes, and the dataset
    is tagged ragged=True so that load_blocks can rebuild them.
    """
    if not isinstance(blocks, list):
        _save_dense(g, name, blocks)
        g[name].attrs["ragged"] = False
        return
    flat = np.concatenate([np.asarray(b).reshape(-1) for b in blocks]) \
        if blocks else np.empty(0)
    shapes = np.array([np.asarray(b).shape for b in blocks], dtype=np.int64) \
        if blocks else np.empty((0, 2), dtype=np.int64)
    _save_dense(g, name, flat)
    g[name].attrs["ragged"] = True
    _save_dense(g, f"{name}_shapes", shapes, compress=False)


def load_blocks(g, name):
    """
    Inverse of _save_blocks. Returns a stacked (N, bs, bs) array for a uniform
    partition and a list of per-block arrays for a ragged one.
    """
    d = g[name]
    if not d.attrs.get("ragged", False):
        return d[:]
    flat = d[:]
    shapes = g[f"{name}_shapes"][:]
    out, pos = [], 0
    for shp in shapes:
        size = int(np.prod(shp))
        out.append(flat[pos:pos + size].reshape(tuple(int(s) for s in shp)))
        pos += size
    return out


def _tag_partition(g, bt):
    """
    Record the block partition on a Block Thomas group.

    block_sizes is always written, so that a group is self-describing whatever
    its partition. block_size is written as an int attribute for uniform
    partitions only, which is what the analysis scripts read to recover the
    block size directly.
    """
    g.attrs["uniform_blocks"] = bool(bt.uniform)
    if bt.bs is not None:
        g.attrs["block_size"] = bt.bs
    _save_dense(g, "block_sizes", np.asarray(bt.block_sizes, dtype=np.int64),
                compress=False)


def _save_common(g, x, t_fact, t_solve=None):
    _save_dense(g, "x", x)
    _save_dense(g, "time_fact", t_fact, compress=False)
    if t_solve is not None:
        _save_dense(g, "time_solve", t_solve, compress=False)


# ---------------------------------------------------------------------------
# Per-solver savers. root is an open h5py.File in mode "a" or "r+", dtype is a
# NumPy dtype, and idx is the energy index.
# ---------------------------------------------------------------------------
def save_superlu(root, dtype, idx, slu, x, t_fact, t_solve=None, mem=None):
    dname = np.dtype(dtype).name
    g = _fresh_group(root, _solver_group_path(idx, "superlu", dname))
    L, U, perm_r, perm_c = slu.get_LUP()
    _save_sparse_factor(g, "L", L)
    _save_sparse_factor(g, "U", U)
    _save_dense(g, "perm_r", perm_r)
    _save_dense(g, "perm_c", perm_c)
    g.attrs["dtype"] = dname
    g.attrs["convention"] = "Pr @ A @ Pc == L @ U  (Pr from argsort(perm_r))"
    if mem is not None:
        g.attrs["factor_nbytes"] = mem
    _save_common(g, x, t_fact, t_solve)


def save_umfpack(root, dtype, idx, umf, x, t_fact, t_solve=None, mem=None):
    """Write a UMFPACK result, including the row scaling R it applies."""
    dname = np.dtype(dtype).name
    g = _fresh_group(root, _solver_group_path(idx, "umfpack", dname))
    L, U, perm_r, perm_c = umf.get_LUP()
    _save_sparse_factor(g, "L", L)
    _save_sparse_factor(g, "U", U)
    _save_dense(g, "perm_r", perm_r)
    _save_dense(g, "perm_c", perm_c)
    R, do_recip = umf.get_scaling()
    if R is not None:
        _save_dense(g, "R", np.asarray(R))
    if do_recip is not None:
        _save_dense(g, "do_recip", np.asarray(do_recip), compress=False)
    g.attrs["dtype"] = dname
    g.attrs["convention"] = ("Pr @ diag(1/R) @ A @ Pc == L @ U  "
                             "(Pr from argsort(perm_r))")
    if mem is not None:
        g.attrs["factor_nbytes"] = mem
    _save_common(g, x, t_fact, t_solve)


def save_blockthomas(root, dtype, idx, bt, x, t_fact, t_solve=None, mem=None,
                     dname=None, group="blockthomas"):
    """
    Write an implementation 1 factorization: L, U, the packed LU of each
    modified diagonal block, and its pivots.

    dname overrides the dtype label in the group path. BlockThomasFP16 stores
    embedded-real float16 factors at block size 2m, for which no complex dtype
    name exists, so the half-precision path passes dname="complex32".
    """
    dname = dname or np.dtype(dtype).name
    g = _fresh_group(root, _solver_group_path(idx, group, dname))
    Lb, Ub, Dlu, Dpiv = bt.get_LUP()
    _save_blocks(g, "L", Lb)
    _save_blocks(g, "U", Ub)
    _save_blocks(g, "Dmod_lu", Dlu)
    _save_blocks(g, "Dmod_piv", Dpiv)
    g.attrs["dtype"] = dname
    g.attrs["implementation"] = 1
    _tag_partition(g, bt)
    if getattr(bt, "s", None) is not None:      # half precision: factors of s*A
        g.attrs["scale_s"] = float(bt.s)
        g.attrs["embedded_real"] = True
    if mem is not None:
        g.attrs["factor_nbytes"] = mem
    _save_common(g, x, t_fact, t_solve)


def save_blockthomas_inv(root, dtype, idx, bt, x, t_fact, t_solve=None, mem=None,
                         dname=None, group="blockthomas_inv"):
    """
    Write an implementation 2 factorization: the explicit block inverses, plus
    D_mod, which the growth-factor analysis requires in order to assemble the
    global U. There are no LU factors to store, since get_LUP() returns None.

    The half-precision variant additionally carries the per-block power-of-two
    scales t, so that the stored inverse of block k is G[k] / t[k] in the
    s-scaled embedded-real space. Both scales are recorded.
    """
    dname = dname or np.dtype(dtype).name
    g = _fresh_group(root, _solver_group_path(idx, group, dname))
    parts = bt.get_inverses()
    if len(parts) == 5:                          # half precision: G, t, D_mod, L, U
        Dinv, t, Dmod, Lb, Ub = parts
        _save_dense(g, "inv_scale_t", np.asarray(t), compress=False)
    else:                                        # complex: Dinv, D_mod, L, U
        Dinv, Dmod, Lb, Ub = parts
    _save_blocks(g, "Dmod_inv", Dinv)
    _save_blocks(g, "Dmod", Dmod)
    _save_blocks(g, "L", Lb)
    _save_blocks(g, "U", Ub)
    g.attrs["dtype"] = dname
    g.attrs["implementation"] = 2
    _tag_partition(g, bt)
    if getattr(bt, "s", None) is not None:
        g.attrs["scale_s"] = float(bt.s)
        g.attrs["embedded_real"] = True
    if getattr(bt, "inv_dtype", None) is not None:
        g.attrs["inv_dtype"] = np.dtype(bt.inv_dtype).name
    if mem is not None:
        g.attrs["factor_nbytes"] = mem
    _save_common(g, x, t_fact, t_solve)


def save_metadata(root, solver_name, dtype, idx, x, t_fact, t_solve=None,
                  metadata=None, mem=None):
    """
    Write a result for a solver that exposes no factors: MUMPS, cuDSS, and the
    iterative solvers. Stores x, the two timings, and any auxiliary arrays or
    scalars supplied in metadata; entries whose value is None are skipped.
    """
    dname = np.dtype(dtype).name
    g = _fresh_group(root, _solver_group_path(idx, solver_name, dname))
    g.attrs["dtype"] = dname
    if mem is not None:
        g.attrs["factor_nbytes"] = mem
    if metadata:
        for key, val in metadata.items():
            if val is None:
                continue
            _save_dense(g, key, np.asarray(val))
    _save_common(g, x, t_fact, t_solve)


# ---------------------------------------------------------------------------
# Factor verification
# ---------------------------------------------------------------------------
def _permutation_matrix(perm):
    """Sparse permutation matrix P with perm[i] = j meaning i maps to j."""
    n = len(perm)
    return sp.csr_matrix((np.ones(n), (np.arange(n), perm)), shape=(n, n))


def verify_superlu_factors(A, L, U, perm_r, perm_c):
    """
    Check the SuperLU convention Pr A Pc == L U, with Pr built from the inverse
    of perm_r through argsort and Pc built directly from perm_c.

    Returns the largest absolute entry of the residual, which is of the order
    of the unit roundoff for a correct factorization.
    """
    inv_perm_r = np.argsort(perm_r)
    Pr = _permutation_matrix(inv_perm_r)
    Pc = _permutation_matrix(perm_c)

    lhs = (Pr @ A @ Pc).tocsr()
    rhs = (L @ U).tocsr()
    diff = (lhs - rhs)
    diff.eliminate_zeros()
    return np.abs(diff.data).max() if diff.nnz else 0.0


def verify_umfpack_factors(A, L, U, perm_r, perm_c, R=None):
    """
    Check the UMFPACK convention Pr diag(1/R) A Pc == L U, with Pr built from
    the inverse of perm_r through argsort, Pc built directly from perm_c, and
    the rows of A scaled by 1/R before permutation. The convention was
    determined empirically and holds to a residual of order 1e-15.

    With R None, meaning no row scaling was applied, this reduces to the
    SuperLU convention.

    Returns the largest absolute entry of the residual.
    """
    inv_perm_r = np.argsort(perm_r)
    Pr = _permutation_matrix(inv_perm_r)
    Pc = _permutation_matrix(perm_c)

    scaled_A = sp.diags(1.0 / R) @ A if R is not None else A

    lhs = (Pr @ scaled_A @ Pc).tocsr()
    rhs = (L @ U).tocsr()
    diff = (lhs - rhs)
    diff.eliminate_zeros()
    return np.abs(diff.data).max() if diff.nnz else 0.0


def diagnose_lu_convention(A, L, U, perm_r, perm_c, R=None, do_recip=None):
    """
    Determine a solver's factor reconstruction convention empirically.

    Enumerates the plausible combinations, row and column permutation taken
    directly or inverted, with and without row scaling, and the scaling or its
    reciprocal, and returns the (description, residual) of the combination with
    the smallest residual.

    Intended to be run once per solver and version, so that the convention is
    established rather than assumed to match SuperLU's.
    """
    rhs = (L @ U).tocsr()

    perm_options = {
        "perm_r direct": perm_r,
        "perm_r inverse (argsort)": np.argsort(perm_r),
    }
    col_options = {
        "perm_c direct": perm_c,
        "perm_c inverse (argsort)": np.argsort(perm_c),
    }

    if R is not None:
        scale_options = {
            "no scaling": None,
            "scale rows by R": sp.diags(R),
            "scale rows by 1/R": sp.diags(1.0 / R),
        }
    else:
        scale_options = {"no scaling": None}

    results = []
    for pr_name, pr_vec in perm_options.items():
        Pr = _permutation_matrix(pr_vec)
        for pc_name, pc_vec in col_options.items():
            Pc = _permutation_matrix(pc_vec)
            for scale_name, D in scale_options.items():
                scaled_A = (D @ A) if D is not None else A
                lhs = (Pr @ scaled_A @ Pc).tocsr()
                diff = (lhs - rhs)
                diff.eliminate_zeros()
                residual = np.abs(diff.data).max() if diff.nnz else 0.0
                results.append((residual, f"{pr_name} | {pc_name} | {scale_name}"))

    results.sort(key=lambda r: r[0])
    best_residual, best_desc = results[0]
    print(f"Best convention found: {best_desc} -> residual {best_residual:.3e}")
    print("Top 3 candidates:")
    for residual, desc in results[:3]:
        print(f"    {residual:.3e}  {desc}")
    return best_desc, best_residual


def load_and_verify(h5group, A, solver="superlu"):
    """
    Reload the factors from an open solver group, for example
    f['E_0/superlu/complex128'], and return their reconstruction residual
    against the original matrix A.
    """
    L = load_sparse_factor(h5group["L"])
    U = load_sparse_factor(h5group["U"])
    perm_r = h5group["perm_r"][:]
    perm_c = h5group["perm_c"][:]
    if solver == "umfpack":
        R = h5group["R"][:] if "R" in h5group else None
        return verify_umfpack_factors(A, L, U, perm_r, perm_c, R=R)
    return verify_superlu_factors(A, L, U, perm_r, perm_c)