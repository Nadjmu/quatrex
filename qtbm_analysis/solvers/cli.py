"""
Canonical names and shared command-line arguments.

Purpose
-------
Single source of truth for two things that were previously spelled differently
in every script: the name of a solver, and the name of a command-line option.
Every executable script in this project builds its parser from the helpers
here, so an option means the same thing and is spelled the same way wherever it
appears.

Solver names
------------
The canonical form is lower-case kebab, listed in SOLVERS. It is used on the
command line, in the `solvers` argument of bench(), in mpir.SOLVER_BUILDERS,
and in the Arnoldi backend list:

    superlu, umfpack, mumps, gmres, gmres-cupy, cudss,
    block-thomas, block-thomas-inv, block-thomas-fp16, block-thomas-inv-fp16

HDF5 group names are deliberately NOT changed
---------------------------------------------
The group names inside a material file (``blockthomas``, ``blockthomas_inv``,
``gmres_scipy``) are on-disk data. Renaming them would make every material file
already written unreadable by the analysis scripts, for no numerical benefit.
They are therefore left as they are, and the mapping between the canonical name
and the stored group is held in SOLVERS and applied by h5_group() and
from_h5_group(). Nothing outside this module needs to know the stored spelling.

Precision names
---------------
Complex working precisions are spelled in full everywhere: complex128,
complex64, and complex32, the storage label for the half-precision
embedded-real factorizations. complex32 is not a NumPy dtype.

The precision in which implementation 2 forms its explicit inverses is a real
dtype, not a complex one, and is spelled float64, float32 or float16.

Option vocabulary
-----------------
    h5path              positional; a material HDF5 file
    matrix, rhs         positional; a CSR .npz triplet and a .npy vector
    --idx N [N ...]     explicit energy indices
    --start / --end     an inclusive index range, the alternative to --idx
    --solver            exactly one solver
    --solvers           one or more solvers
    --dtypes            one or more complex working precisions
    --factor-dtype      the precision of a factorization, u_f
    --inv-dtype         the precision in which an explicit inverse is formed
    --block-size        a uniform Block Thomas block size
    --auto-blocks       detect a non-uniform partition from the sparsity pattern
    --material          label used in output filenames and figure titles
    --outdir            output directory
"""

import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Solver registry
# ---------------------------------------------------------------------------
# Per canonical name:
#   label      human-readable name, used in reports and figure legends
#   h5         the HDF5 group name under E_<idx>/, or None where the solver
#              stores nothing of its own
#   gpu        requires a visible CUDA device
#   dtypes     complex working precisions the solver accepts; the
#              half-precision variants are precision-fixed and carry the
#              storage label complex32
#   factors    whether explicit factors are exposed
SOLVERS = {
    "superlu": dict(label="SuperLU", h5="superlu", gpu=False,
                    dtypes=("complex128", "complex64"), factors=True),
    "umfpack": dict(label="UMFPACK", h5="umfpack", gpu=False,
                    dtypes=("complex128",), factors=True),
    "mumps": dict(label="MUMPS", h5="mumps", gpu=False,
                  dtypes=("complex128", "complex64"), factors=False),
    "gmres": dict(label="GMRES (SciPy)", h5="gmres_scipy", gpu=False,
                  dtypes=("complex128", "complex64"), factors=False),
    "gmres-cupy": dict(label="GMRES (CuPy)", h5="gmres_cupy", gpu=True,
                       dtypes=("complex128", "complex64"), factors=False),
    "cudss": dict(label="cuDSS", h5="cudss", gpu=True,
                  dtypes=("complex128", "complex64"), factors=False),
    "block-thomas": dict(label="Block Thomas (LU)", h5="blockthomas",
                         gpu=False, dtypes=("complex128", "complex64"),
                         factors=True),
    "block-thomas-inv": dict(label="Block Thomas (inv)", h5="blockthomas_inv",
                             gpu=False, dtypes=("complex128", "complex64"),
                             factors=True),
    "block-thomas-fp16": dict(label="Block Thomas fp16 (LU)", h5="blockthomas",
                              gpu=False, dtypes=("complex32",), factors=True),
    "block-thomas-inv-fp16": dict(label="Block Thomas fp16 (inv)",
                                  h5="blockthomas_inv", gpu=False,
                                  dtypes=("complex32",), factors=True),
}

ALL_SOLVERS = tuple(SOLVERS)

# Solvers benchmarked by default: everything precision-parametric. The
# half-precision variants are excluded because their NumPy kernels are orders
# of magnitude slower than LAPACK; request them explicitly.
DEFAULT_SOLVERS = ("superlu", "umfpack", "mumps", "gmres", "gmres-cupy",
                   "cudss", "block-thomas", "block-thomas-inv")

FP16_SOLVERS = ("block-thomas-fp16", "block-thomas-inv-fp16")

BLOCK_SOLVERS = ("block-thomas", "block-thomas-inv",
                 "block-thomas-fp16", "block-thomas-inv-fp16")

GPU_SOLVERS = tuple(n for n, m in SOLVERS.items() if m["gpu"])

# Solvers that store explicit factors, hence can be analysed for factor growth.
FACTOR_SOLVERS = ("block-thomas", "block-thomas-inv", "superlu", "umfpack")

# ---------------------------------------------------------------------------
# Precision names
# ---------------------------------------------------------------------------
# complex32 is a storage label, not a NumPy dtype: np.dtype("complex32") does
# not exist. It denotes the half-precision embedded-real factorizations.
COMPLEX_DTYPES = ("complex128", "complex64", "complex32")
WORKING_DTYPES = ("complex128", "complex64")
INV_DTYPES = ("float64", "float32", "float16")

DEFAULT_DTYPES = ("complex128", "complex64")

# Short suffix used in the in-memory result keys of bench().
DTYPE_SUFFIX = {"complex128": "c128", "complex64": "c64",
                "complex32": "fp16", "float64": "f64", "float32": "f32",
                "float16": "f16"}


def np_dtype(name):
    """NumPy dtype for a precision name; None for the complex32 label."""
    return None if name == "complex32" else np.dtype(name)


def dtype_suffix(name):
    """Short suffix for a precision name, as used in bench() result keys."""
    name = np.dtype(name).name if not isinstance(name, str) else name
    return DTYPE_SUFFIX.get(name, name)


# ---------------------------------------------------------------------------
# Canonical name to stored HDF5 group name
# ---------------------------------------------------------------------------
def h5_group(solver):
    """
    HDF5 group name under E_<idx>/ for a canonical solver name.

    The two half-precision variants share the group of their complex
    counterparts and are distinguished by the complex32 precision level
    beneath it, which is how factor_io has always written them.
    """
    try:
        return SOLVERS[solver]["h5"]
    except KeyError:
        raise KeyError(f"unknown solver {solver!r}; "
                       f"choose from {', '.join(ALL_SOLVERS)}") from None


def from_h5_group(group, dtype=None):
    """
    Canonical solver name for a stored HDF5 group name.

    `dtype` disambiguates the two Block Thomas implementations, whose
    half-precision variants share a group with their complex counterparts:
    passing "complex32" selects the half-precision name.
    """
    for name, meta in SOLVERS.items():
        if meta["h5"] != group:
            continue
        is_fp16 = name.endswith("-fp16")
        if (dtype == "complex32") == is_fp16:
            return name
    return group


def label(solver):
    """Human-readable label for a canonical solver name."""
    return SOLVERS.get(solver, {}).get("label", solver)


def supports(solver, dtype):
    """True if `solver` accepts the precision `dtype`."""
    return dtype in SOLVERS.get(solver, {}).get("dtypes", ())


# ---------------------------------------------------------------------------
# Shared argparse fragments
# ---------------------------------------------------------------------------
def new_parser(description, epilog=None, prog=None):
    """Parser with the conventions this project uses for --help rendering."""
    return argparse.ArgumentParser(
        prog=prog, description=description, epilog=epilog,
        formatter_class=argparse.RawDescriptionHelpFormatter)


def add_h5_input(ap, required=True, default=None):
    """Positional material HDF5 file, spelled `h5path` in every script."""
    ap.add_argument("h5path", type=str, nargs=None if required else "?",
                    default=default, help="material HDF5 file")
    return ap


def add_index_selection(ap, default_all=True):
    """
    Energy index selection: --idx for explicit indices, --start and --end for
    an inclusive range. The two are mutually exclusive.
    """
    group = ap.add_mutually_exclusive_group(required=not default_all)
    group.add_argument("--idx", type=int, nargs="+", default=None,
                       metavar="N",
                       help="one or more explicit energy indices")
    group.add_argument("--start", type=int, default=None, metavar="N",
                       help="first energy index of an inclusive range; "
                            "requires --end")
    ap.add_argument("--end", type=int, default=None, metavar="N",
                    help="last energy index of the range, inclusive")
    return ap


def resolve_indices(ap, args, available=None):
    """
    Index list implied by --idx or --start/--end.

    With neither given, returns `available` if it was supplied, otherwise
    raises a parser error. Indices absent from `available` are reported and
    dropped rather than causing a failure part-way through a sweep.
    """
    if args.idx is not None:
        requested = list(args.idx)
    elif args.start is not None:
        if args.end is None:
            ap.error("--start requires --end")
        requested = list(range(args.start, args.end + 1))
    elif available is not None:
        return list(available)
    else:
        ap.error("give either --idx N [N ...] or --start S --end E")

    if available is None:
        return requested

    available = set(available)
    selected = [i for i in requested if i in available]
    missing = [i for i in requested if i not in available]
    if missing:
        shown = missing[:20]
        more = " ..." if len(missing) > 20 else ""
        print(f"[warning] requested indices not present: {shown}{more}")
    return selected


def add_solver_selection(ap, choices=ALL_SOLVERS, default=None, multiple=True,
                         help=None):
    """--solvers (one or more) or --solver (exactly one), canonical names."""
    if multiple:
        ap.add_argument("--solvers", nargs="+", choices=list(choices),
                        default=list(default) if default else None,
                        metavar="NAME",
                        help=help or f"solvers to run; choose from "
                                     f"{', '.join(choices)}")
    else:
        ap.add_argument("--solver", choices=list(choices), default=default,
                        metavar="NAME",
                        help=help or f"solver to use; choose from "
                                     f"{', '.join(choices)}")
    return ap


def add_dtypes(ap, choices=WORKING_DTYPES, default=DEFAULT_DTYPES, help=None):
    """--dtypes, one or more complex working precisions."""
    ap.add_argument("--dtypes", nargs="+", choices=list(choices),
                    default=list(default) if default else None,
                    metavar="NAME",
                    help=help or "complex working precisions; "
                                 f"choose from {', '.join(choices)}")
    return ap


def add_factor_dtype(ap, choices=WORKING_DTYPES, default="complex128",
                     help=None):
    """--factor-dtype, the precision of a factorization."""
    ap.add_argument("--factor-dtype", choices=list(choices), default=default,
                    metavar="NAME",
                    help=help or f"precision of the factorization "
                                 f"(default: {default})")
    return ap


def add_inv_dtype(ap, default="float32"):
    """--inv-dtype, the precision in which an explicit inverse is formed."""
    ap.add_argument("--inv-dtype", choices=list(INV_DTYPES), default=default,
                    metavar="NAME",
                    help=f"precision in which the half-precision "
                         f"explicit-inverse variant forms its inverses before "
                         f"rounding them to float16 (default: {default}). "
                         f"float16 makes the factorization half precision "
                         f"throughout")
    return ap


def add_block_partition(ap, default_block_size=None, auto=True):
    """--block-size, and optionally --auto-blocks."""
    ap.add_argument("--block-size", type=int, default=default_block_size,
                    metavar="M",
                    help="uniform Block Thomas block size"
                         + ("" if default_block_size is None
                            else f" (default: {default_block_size})"))
    if auto:
        ap.add_argument("--auto-blocks", action="store_true",
                        help="detect a non-uniform partition from the "
                             "sparsity pattern instead of using --block-size")
    return ap


def resolve_partition(args):
    """
    Partition implied by --block-size and --auto-blocks.

    Returns None when --auto-blocks was given, meaning the caller must detect
    the partition from the matrix, and the integer block size otherwise.
    """
    if getattr(args, "auto_blocks", False):
        return None
    return args.block_size


def add_output(ap, material=True, outdir_default=None, outdir_help=None):
    """--outdir, and optionally --material."""
    ap.add_argument("--outdir", type=str, default=outdir_default,
                    metavar="DIR",
                    help=outdir_help or "output directory")
    if material:
        ap.add_argument("--material", type=str, default=None, metavar="NAME",
                        help="label used in output filenames and figure "
                             "titles (default: derived from the input path)")
    return ap
