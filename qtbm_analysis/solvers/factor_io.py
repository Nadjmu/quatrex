"""
factor_io.py -- HDF5 persistence for solver factors / metadata.

Solver output is appended directly into the MATERIAL's own HDF5 file
(the same file that has E_<idx>/M, E_<idx>/rhs, etc.) -- there is no
separate *_LU.h5 file. Each solver's group is a DIRECT sibling of
M/Sigma/rhs/spectrum inside its energy index's own group -- e.g.
E_<idx>/superlu sits next to E_<idx>/M, not nested under it:

    E_<idx>/superlu/<dtype>/       L, U (sparse CSC), perm_r, perm_c, x, time_fact, time_solve
    E_<idx>/umfpack/<dtype>/       L, U, perm_r, perm_c, R, do_recip, x, time_fact, time_solve
    E_<idx>/blockthomas/<dtype>/   L, U (stacked dense blocks), Dmod_lu, Dmod_piv, x, time_fact, time_solve
    E_<idx>/mumps/<dtype>/         x, time_fact, time_solve          (no factors exposed)
    E_<idx>/cudss/<dtype>/         col/row_permutation, lu_nnz, x, time_fact, time_solve
    E_<idx>/gmres*/<dtype>/        x, time_fact, time_solve, iters, info  (iterative -- no factors)

Every group above also carries a "factor_nbytes" attribute (best-effort
memory footprint of the stored factors, from each solver's factor_nbytes())
whenever a saver is called with mem=<int>.

`root` passed to every save_* / load_* function here must be the material's
own open HDF5 file (mode "a" or "r+"), NOT a separate output file -- pass
h5py.File(h5path, "a") where h5path is the same path used to read M/rhs.

Each save_* call DELETES and recreates the E_<idx>/<solver>/<dtype> group
first, so re-running a (solver, dtype, idx) combination overwrites cleanly
-- no stale datasets or attrs left behind, and E_<idx>/M, /rhs, /Sigma,
/spectrum are untouched since save_* never touches anything outside its
own E_<idx>/<solver>/<dtype> group.

Reconstruction conventions (stored as a group attribute too):
    superlu :  Pr @ A @ Pc == L @ U               (Pr from argsort(perm_r))
    umfpack :  Pr @ diag(1/R) @ A @ Pc == L @ U   (Pr from argsort(perm_r))
"""

import numpy as np
import scipy.sparse as sp


# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------
def _fresh_group(root, group_path):
    """Delete group_path if it exists, then (re)create it empty."""
    if group_path in root:
        del root[group_path]
    return root.require_group(group_path)


def _solver_group_path(idx, solver_name, dname):
    """
    Path for one (solver, dtype, energy-index) result, nested INSIDE that
    energy index's own group so solver output lives alongside the matrix
    that produced it:

        E_<idx>/<solver_name>/<dtype>/...

    i.e. superlu, umfpack, etc. sit as DIRECT siblings of E_<idx>/M,
    E_<idx>/rhs, E_<idx>/Sigma -- not nested under an extra "solvers" level,
    and not a separate top-level tree. `root` must therefore be the
    material's own HDF5 file (opened "a"/"r+"), the same file E_<idx>/M
    lives in, not a separate *_LU.h5 file.
    """
    return f"E_{int(idx)}/{solver_name}/{dname}"


def _save_sparse_factor(g, name, mat):
    """Save a scipy sparse (CSC) matrix under group g/name (overwrite)."""
    sg = g.require_group(name)
    for k in list(sg.keys()):
        del sg[k]
    mat = mat.tocsc()
    sg.create_dataset("data",    data=mat.data,    compression="gzip")
    sg.create_dataset("indices", data=mat.indices, compression="gzip")
    sg.create_dataset("indptr",  data=mat.indptr,  compression="gzip")
    sg.attrs["shape"] = mat.shape


def load_sparse_factor(g):
    """Inverse of _save_sparse_factor: group with data/indices/indptr -> CSC."""
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


def _save_common(g, x, t_fact, t_solve=None):
    _save_dense(g, "x", x)
    _save_dense(g, "time_fact", t_fact, compress=False)
    if t_solve is not None:
        _save_dense(g, "time_solve", t_solve, compress=False)


# ---------------------------------------------------------------------------
# per-solver savers  (root = open h5py.File, dtype = np dtype, idx = energy idx)
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
    """umf is a solver_classes.UMFPACK instance."""
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


def save_blockthomas(root, dtype, idx, bt, x, t_fact, t_solve=None, mem=None):
    dname = np.dtype(dtype).name
    g = _fresh_group(root, _solver_group_path(idx, "blockthomas", dname))
    Lb, Ub, Dlu, Dpiv = bt.get_LUP()
    _save_dense(g, "L", Lb)
    _save_dense(g, "U", Ub)
    _save_dense(g, "Dmod_lu", Dlu)
    _save_dense(g, "Dmod_piv", Dpiv)
    g.attrs["dtype"] = dname
    g.attrs["block_size"] = bt.bs
    if mem is not None:
        g.attrs["factor_nbytes"] = mem
    _save_common(g, x, t_fact, t_solve)


def save_metadata(root, solver_name, dtype, idx, x, t_fact, t_solve=None,
                  metadata=None, mem=None):
    """
    For solvers with no explicit factors (MUMPS, cuDSS) and for iterative
    solvers: saves x, factor/solve time, and whatever auxiliary arrays or
    scalars are handed in via `metadata` (dict; None values skipped).
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
# Factor verification (ported from the old helpers.py)
# ---------------------------------------------------------------------------
def _permutation_matrix(perm):
    """perm[i] = j means row/col i maps to position j -> build as sparse matrix."""
    n = len(perm)
    return sp.csr_matrix((np.ones(n), (np.arange(n), perm)), shape=(n, n))


def verify_superlu_factors(A, L, U, perm_r, perm_c):
    """
    Checks the SciPy SuperLU convention:  Pr @ A @ Pc == L @ U
    where Pr is built from the *inverse* of perm_r (argsort), and Pc is
    built directly from perm_c. Returns the max absolute residual entry
    (should be ~machine epsilon for a correct factorization).
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
    Confirmed UMFPACK convention (empirically determined, residual ~1e-15):

        Pr @ diag(1/R) @ A @ Pc == L @ U

    where Pr is built from the *inverse* of perm_r (argsort), Pc is built
    directly from perm_c, and rows of A are scaled by 1/R before permuting.
    If R is None (no row scaling was used), this reduces to the plain
    SuperLU-style convention.
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
    Brute-forces plausible LU reconstruction conventions (row/col permutation
    direct vs. inverse, with/without row scaling, scaling vs. its reciprocal)
    and returns the (description, residual) of whichever combination gives
    the smallest residual. Use this once per solver/version to pin down the
    exact convention empirically, instead of assuming it matches SuperLU's.
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
    Convenience: given an open h5 group like f['E_0/superlu/complex128']
    (or the umfpack equivalent) and the original matrix A, reload the
    factors and return the reconstruction residual.
    """
    L = load_sparse_factor(h5group["L"])
    U = load_sparse_factor(h5group["U"])
    perm_r = h5group["perm_r"][:]
    perm_c = h5group["perm_c"][:]
    if solver == "umfpack":
        R = h5group["R"][:] if "R" in h5group else None
        return verify_umfpack_factors(A, L, U, perm_r, perm_c, R=R)
    return verify_superlu_factors(A, L, U, perm_r, perm_c)