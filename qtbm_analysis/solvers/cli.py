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
    h5path              positional; a material HDF5 file, or an analysis file
                        for the plotting scripts
    matrix, rhs         positional; a CSR .npz triplet and a .npy vector
    --idx N [N ...]     explicit energy indices
    --start / --end     an inclusive index range, the alternative to --idx
    --stride            keep every Nth index of the selection above
    --solver            exactly one solver
    --solvers           one or more solvers
    --dtypes            one or more complex working precisions
    --factor-dtype      the precision of a factorization, u_f
    --inv-dtype         the precision in which an explicit inverse is formed
    --material          label used in output filenames and figure titles
    --outdir            output directory

Scratch layout
--------------
The default location of every input and output is defined below and nowhere
else. Each analysis stage has one directory, holding both its result files and
the figures made from them. Set QTBM_SCRATCH to relocate the whole tree, for
example when running outside the cluster; every path derives from it.

Materials
---------
MATERIALS is the one place a per-material property is edited: the band edges,
the energy grid, the Block Thomas block size, the input directories, and the
parameters the contact band structure needs.

The grid of each material is an explicit EnergyGrid(start, end, resolution) in
eV. It is not derived from the band edges, and the number of energy indices is
not stated anywhere; it follows from the three numbers.
"""

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scratch layout
# ---------------------------------------------------------------------------
# One directory per stage of the pipeline and one per analysis. The analysis
# directories are named after the thesis chapter whose figures they hold, so
# the mapping from a result file to the text that uses it is direct.
SCRATCH = Path(os.environ.get("QTBM_SCRATCH", "/scratch/yimili"))

EXAMPLES_DIR = SCRATCH / "examples"           # QTBM/DFT inputs, stage 1 source
EXPORT_DIR = SCRATCH / "matrices2"            # stage 1: per-material .npz/.npy
HDF5_DIR = EXPORT_DIR / "hdf5"                # stage 2/3: <material>.h5
RANDOM_DIR = SCRATCH / "random"               # synthetic test matrices

# Stage 4/5. Each holds <material>.h5 and the figures drawn from it.
BLOCK_THOMAS_DIR = SCRATCH / "error-analysis-block-thomas"
CONDITION_DIR = SCRATCH / "condition-est"
NON_NORMAL_DIR = SCRATCH / "non-normal"
MIXED_PREC_DIR = SCRATCH / "mixed-precision-IR"
MATERIALS_DIR = SCRATCH / "materials"


def material_h5(material):
    """Path of a material file, the input of every stage-3 and stage-4 script."""
    return HDF5_DIR / f"{material}.h5"


def analysis_h5(outdir, material):
    """
    Path of an analysis result file inside one stage-4 directory.

    One file per material per directory; each script writes its own top-level
    group into it, so results of different analyses of the same material stay
    together and neither overwrites the other.
    """
    return Path(outdir) / f"{material}.h5"


# ---------------------------------------------------------------------------
# Energy grid
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyGrid:
    """
    The energy sweep of one material: a first energy, a last energy and a step,
    all in eV.

    The grid is `start`, `start + resolution`, ... up to and including `end`.
    The number of energy indices follows from the three numbers and is stated
    nowhere else: halving `resolution` doubles it, and every stage of the
    pipeline picks the change up. Nothing downstream assumes a particular
    count.

    `end` is included only if it is an exact number of steps above `start`.
    Where it is not, the grid stops at the last sample below it and `last` is
    that sample rather than `end`; the two are reported separately so the
    difference is visible.
    """

    start: float
    end: float
    resolution: float

    def __post_init__(self):
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")
        if self.end <= self.start:
            raise ValueError(f"end ({self.end}) must exceed start ({self.start})")

    @property
    def num_points(self):
        """Number of samples, hence of energy indices."""
        steps = (self.end - self.start) / self.resolution
        # Tolerate the rounding of a step that does not divide the span
        # exactly, without ever running past `end`.
        return int(np.floor(steps + 1e-9)) + 1

    @property
    def last(self):
        """Energy of the final sample, which is at most `end`."""
        return self.start + self.resolution * (self.num_points - 1)

    def energies(self):
        """
        The grid itself, as an array of `num_points` energies in eV.

        This is the only definition of the grid in the project. Stage 1 saves
        the array it produces and stage 2 reads that array back, so the two
        cannot disagree even if these parameters are changed between runs.
        """
        return self.start + self.resolution * np.arange(self.num_points,
                                                        dtype=float)

    def index_of(self, energy):
        """
        Fractional index of an energy, for placing a mark on an index axis.

        Outside [start, last] the value is still returned, so a caller can
        decide whether a band edge lies inside the sweep.
        """
        return (float(energy) - self.start) / self.resolution

    @classmethod
    def around(cls, centre, half_window, resolution, offset_low=0.0):
        """
        A grid centred on `centre`, from `centre - half_window - offset_low` to
        `centre + half_window`.

        Provided because a sweep is often described relative to a band edge.
        The stored form is still start, end and resolution.
        """
        return cls(start=centre - half_window - offset_low,
                   end=centre + half_window, resolution=resolution)


# ---------------------------------------------------------------------------
# Materials
# ---------------------------------------------------------------------------
@dataclass
class Material:
    """
    Everything known about one material, in one place.

    Fields
    ------
    example
        QTBM example directory holding quatrex_config.toml. Stage 1 assembles
        M(E) from it.
    inputs
        DFT input directory holding hamiltonian.mat and, for a non-orthogonal
        basis, overlap.mat. plotting/materials/bandstructure.py reads it.
    block_size
        Uniform Block Thomas block size. One periodic lead block.
    blocks
        Non-uniform partition, if one is used in place of block_size.
    valence_band_edge, conduction_band_edge
        In eV. Written into the material file's metadata and marked by every
        figure. None means not yet determined: stage 1 then falls back to the
        values in the QTBM configuration, and stage 2 records only what it has.
    grid
        The energy sweep, as an explicit EnergyGrid(start, end, resolution) in
        eV. It is not derived from the band edges; set it to whatever range and
        resolution the material is to be swept over.
    mid_gap_energy
        An energy inside the band gap, used by plotting/materials/bandstructure.py to
        separate the valence from the conduction bands when it locates the
        edges. Not used elsewhere.
    transverse_k, num_k_points, lead_offset
        Contact band structure parameters; see plotting/materials/bandstructure.py.
    """

    example: Path | None = None
    inputs: Path | None = None
    block_size: int | None = None
    blocks: list | None = None
    valence_band_edge: float | None = None
    conduction_band_edge: float | None = None
    grid: EnergyGrid | None = None
    mid_gap_energy: float | None = None
    transverse_k: tuple = (0.0, 0.0, 0.0)
    num_k_points: int = 401
    lead_offset: int = 0

    @property
    def band_gap(self):
        """Conduction minus valence edge, or None if either is unknown."""
        if self.valence_band_edge is None or self.conduction_band_edge is None:
            return None
        return self.conduction_band_edge - self.valence_band_edge


# Edit the band edges and the grid here. The grid entries below reproduce the
# sweep used so far, [conduction edge - 1.005, conduction edge + 1.0] at 0.005
# eV, written out as explicit energies; replace the numbers with the range and
# resolution each material is to be swept over. The index count follows.
MATERIALS = {
    "carbon-nanotube": Material(
        example=EXAMPLES_DIR / "w90/carbon-nanotube/qtbm",
        inputs=EXAMPLES_DIR / "w90/carbon-nanotube/inputs",
        block_size=32,
        valence_band_edge=-4.17,
        conduction_band_edge=-3.57,
        grid=EnergyGrid(start=-5.17, end=-2.57, resolution=0.001),
        mid_gap_energy=-3.87,
    ),
    "si-bulk": Material(
        # Bandwidth 382 -> smallest admissible divisor of 3840 is 192.
        example=EXAMPLES_DIR / "w90/si-bulk/qtbm",
        inputs=EXAMPLES_DIR / "w90/si-bulk/inputs",
        block_size=256,
        valence_band_edge=5.69,
        conduction_band_edge=6.46,
        grid=EnergyGrid(start=4.69, end=7.46, resolution=0.001),
        mid_gap_energy=6.075,
    ),
    "carbon-chain": Material(
        example=EXAMPLES_DIR / "cp2k/carbon-chain/qtbm",
        inputs=EXAMPLES_DIR / "cp2k/carbon-chain/inputs",
        block_size=104,
        valence_band_edge=-13.74,
        conduction_band_edge=-9.86,
        grid=EnergyGrid(start=-14.74, end=-8.86, resolution=0.001),
        mid_gap_energy=-11.8,
    ),
    "graphene": Material(
        # Bandwidth 431 -> smallest admissible divisor of 2080 is 260.
        example=EXAMPLES_DIR / "graphene/qtbm",
        inputs=EXAMPLES_DIR / "graphene/inputs",
        block_size=416,
        # The edges are the boundaries of the transmitting region, read off
        # the exported right-hand sides: the last index with open channels
        # below the gap is E_532 (+0.032 eV) and the first above it is E_1615
        # (+1.115 eV). The values the export carries, 0.499 and 0.500, are
        # what find_band_edges returns when it is handed mid_gap_energy=0.5,
        # and they sit in the middle of the 1.08 eV window in which QTBM
        # injects no modes at all. mid_gap_energy is left at 0.5, which is
        # still inside the gap and so still a valid input to bandstructure.py.
        valence_band_edge=0.032,
        conduction_band_edge=1.115,
        grid=EnergyGrid(start=-0.5, end=1.500, resolution=0.001),
        mid_gap_energy=0.5,
    ),
    "ws2-hbn": Material(
        example=EXAMPLES_DIR / "WS2-hBN-25_benchmark-QUATREX-DZ/qtbm",
        inputs=EXAMPLES_DIR / "WS2-hBN-25_benchmark-QUATREX-DZ/qtbm/inputs",
        # Bandwidth 6303 -> 6526 is the smallest divisor that guarantees
        # block tridiagonality.
        block_size=6526,
        mid_gap_energy=None,
        # The blocks are large enough that each eigensolve is expensive.
        num_k_points=21,
    ),
    "dev_12_sorted_BENCH": Material(
        example=EXAMPLES_DIR / "dev_12_sorted_BENCH/qtbm",
    ),
}


def material(name):
    """Registry entry for a material name."""
    try:
        return MATERIALS[name]
    except KeyError:
        raise KeyError(f"unknown material {name!r}; "
                       f"choose from {', '.join(MATERIALS)}") from None


def band_edge_attrs(mat):
    """
    Band edge metadata of a Material, as a dict, omitting unknown values.

    Written into the material file's metadata group by make_hdf5.py and copied
    into every analysis file by the stage-4 scripts, so that a figure can mark
    the edges without opening the material file.
    """
    attrs = {}
    if mat is None:
        return attrs
    if mat.valence_band_edge is not None:
        attrs["valence_band_edge"] = float(mat.valence_band_edge)
    if mat.conduction_band_edge is not None:
        attrs["conduction_band_edge"] = float(mat.conduction_band_edge)
    if mat.band_gap is not None:
        attrs["band_gap"] = float(mat.band_gap)
    return attrs

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


def add_h5_input(ap, required=True, default=None, help=None):
    """
    Positional HDF5 input file, spelled `h5path` in every script.

    A material file for the solver and analysis drivers, and the analysis file
    written by one of them for the plotting scripts; `help` names which.
    """
    ap.add_argument("h5path", type=str, nargs=None if required else "?",
                    default=default, help=help or "material HDF5 file")
    return ap


def add_index_selection(ap, default_all=True):
    """
    Energy index selection: --idx for explicit indices, --start and --end for
    an inclusive range, and --stride to thin any of those. The two index
    sources are mutually exclusive; --stride composes with either, or with
    neither.
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
    ap.add_argument("--stride", type=int, default=1, metavar="N",
                    help="keep every Nth index of the selection, in "
                         "ascending order (default: 1, every index). The "
                         "index values kept are unchanged -- E_4 is still "
                         "E_4 at --stride 2, not renumbered -- so energy(i) "
                         "= grid_energy_min + resolution * i still holds for "
                         "every kept index, and every downstream script that "
                         "reads an index list from the file needs no change.")
    return ap


def index_of_energy(attrs, energies):
    """
    Nearest energy-grid index for each energy in eV -- the inverse of
    style.energies_of(). `attrs` needs grid_energy_min and resolution, which
    material_metadata() copies into every stage-4 group; raises SystemExit
    otherwise, since there is then no grid to invert.
    """
    if attrs is None or "grid_energy_min" not in attrs or "resolution" not in attrs:
        raise SystemExit(
            "--energy requires grid_energy_min/resolution metadata, absent "
            "from this file"
        )
    start = float(attrs["grid_energy_min"])
    step = float(attrs["resolution"])
    return [int(round((float(e) - start) / step)) for e in energies]


def available_indices(h5_file, require="M"):
    """
    Energy indices actually present in an open material file, ascending.

    `metadata/indices`, the full sweep make_hdf5.py records, is the candidate
    list where the file has one; otherwise the E_<idx> groups themselves are
    scanned. Either way an index is counted only when its group holds every
    name in `require`, a string or a sequence of them, so a partially written
    file reports what it really contains. Pass require=None to accept every
    group.

    The sweeps are thousands of indices long and their extent is a property of
    the file, never of a hard-coded range, so every driver that offers a
    default-all index selection resolves it through this function and passes
    the result to resolve_indices as `available`.
    """
    required = () if require is None else (
        (require,) if isinstance(require, str) else tuple(require))

    if "metadata/indices" in h5_file:
        candidates = [int(i) for i in h5_file["metadata/indices"][:]]
    else:
        candidates = [int(k[2:]) for k in h5_file
                      if k.startswith("E_") and k[2:].isdigit()]

    out = []
    for index in candidates:
        group = h5_file.get(f"E_{index}")
        if group is None or not hasattr(group, "keys"):
            continue
        if all(name in group for name in required):
            out.append(index)
    return sorted(out)


def resolve_indices(ap, args, available=None):
    """
    Index list implied by --idx or --start/--end, thinned by --stride.

    With neither --idx nor --start given, returns `available` if it was
    supplied, otherwise raises a parser error. Indices absent from
    `available` are reported and dropped rather than causing a failure
    part-way through a sweep. --stride is applied last, to the final
    ascending list, so it thins whichever source produced it uniformly.

    Because the kept index values are the literal grid positions rather than
    positions renumbered from 0, a --stride > 1 selection composes freely
    with everything downstream that already keys off the index value: energy
    lookup, band-edge marking, and any later script re-opening the same file
    with its own --idx/--start/--end need no stride awareness of their own.
    """
    if args.idx is not None:
        requested = list(args.idx)
    elif args.start is not None:
        if args.end is None:
            ap.error("--start requires --end")
        requested = list(range(args.start, args.end + 1))
    elif available is not None:
        requested = list(available)
    else:
        ap.error("give either --idx N [N ...] or --start S --end E")

    if available is not None:
        available_set = set(available)
        missing = [i for i in requested if i not in available_set]
        requested = [i for i in requested if i in available_set]
        if missing:
            shown = missing[:20]
            more = " ..." if len(missing) > 20 else ""
            print(f"[warning] requested indices not present: {shown}{more}")

    stride = getattr(args, "stride", 1)
    if stride > 1:
        requested = sorted(requested)[::stride]
    return requested


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


# There is deliberately no --block-size / --auto-blocks pair here. Every driver
# detects the Block Thomas partition from the sparsity pattern with
# solver_classes.block_sizes_from_matrix; the exported matrices have a
# non-uniform block structure, so a uniform partition either cuts a real
# coupling or pads the blocks, and offering it as an option only invites a
# silently wrong result.


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
