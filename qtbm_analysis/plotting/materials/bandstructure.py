"""Contact band structures of the example devices.

For each material the periodic lead blocks (h_00, h_01, h_10) are
extracted from the device Hamiltonian and used to compute the contact
band structure and the band edges.

Output goes to one subdirectory per material inside the materials directory
defined in `solvers/cli.py`.

Usage:
    python bandstructure.py               # all configured materials
    python bandstructure.py mos2 si-bulk  # a subset
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import scipy.sparse as sps

sys.path.append(str((Path(__file__).resolve().parent / ".." / ".."
                     / "solvers").resolve()))
sys.path.append(str((Path(__file__).resolve().parent / "..").resolve()))

import cli
from style import write_data_report

# Above this many band-energy samples the report gives the per-band envelope
# over k rather than the full (k, band) grid.
MAX_BAND_SAMPLES = 60_000


from qttools import xp
from qttools.kernels.linalg import eigvalsh
from qttools.utils.gpu_utils import get_host
from quatrex.bandstructure import band_edges

EXAMPLES_DIR = cli.EXAMPLES_DIR
PLOTS_DIR = cli.MATERIALS_DIR

# Memory budget for one chunk of k points. Lower this if the eigensolve
# runs out of device memory.
K_CHUNK_BYTES = 8 * 1024**3


# The materials this script can process: those of cli.MATERIALS that declare a
# DFT input directory. Every per-material property, the block size and the
# mid-gap energy included, is edited there and not here. A material whose
# blocksize or mid_gap_energy is still None stops with the report that
# determines it.
MATERIALS = {name: mat for name, mat in cli.MATERIALS.items()
             if mat.inputs is not None}


def load_matrices(path: Path) -> dict[str, sps.csr_matrix]:
    """Loads all matrices from a MATLAB file, keyed by neighbor cell.

    The matrices are kept sparse: the device Hamiltonians are far too
    large to densify, only their lead blocks are.

    """
    mat = scipy.io.loadmat(path)
    return {
        k: (v.tocsr() if sps.issparse(v) else sps.csr_matrix(v))
        for k, v in mat.items()
        if not k.startswith("__")
    }


def block(matrix: sps.spmatrix, blocksize: int, row: int, col: int) -> np.ndarray:
    """Extracts one dense block of a sparse matrix."""
    return matrix[
        row * blocksize : (row + 1) * blocksize,
        col * blocksize : (col + 1) * blocksize,
    ].toarray()


def lead_blocks(matrix: sps.spmatrix, blocksize: int, offset: int = 0) -> tuple:
    """Extracts the diagonal and off-diagonal blocks of a periodic lead.

    `offset` selects which block of the device to take them from. The
    first block is not always bulk-like: a device can start on a partial
    cell, in which case its blocks are not a valid periodic lead.

    """
    i = offset
    return (
        xp.asarray(block(matrix, blocksize, i, i)),
        xp.asarray(block(matrix, blocksize, i, i + 1)),
        xp.asarray(block(matrix, blocksize, i + 1, i)),
    )


def report_hopping_grid(matrices: dict[str, sps.spmatrix]) -> None:
    """Reports the magnitude of each neighbor-cell matrix.

    The direction along which the couplings are strongest is a good
    indication of the transport direction.

    """
    keys = sorted(
        matrices, key=lambda k: tuple(int(i) for i in k.strip("[]").split(","))
    )
    print("  neighbor cells:")
    for key in keys:
        print(f"    {key:>14}  max|H| = {abs(matrices[key]).max():.4e}")


def assemble_device(
    matrices: dict[str, np.ndarray], transverse_k: tuple[float, float, float]
) -> np.ndarray:
    """Assembles the device matrix at a given transverse k point.

    The matrices are keyed by neighbor cell as "[x, y, z]" and are summed
    as sum_R M(R) exp(2i pi k.R). Since the device is already assembled
    along the transport direction, these cells are the transverse
    periodic images, and `transverse_k` selects the transverse momentum
    (in fractional coordinates; the origin is Gamma).

    """
    device = 0
    for key, matrix in matrices.items():
        r = [int(i) for i in key.strip("[]").split(",")]
        device = device + np.exp(2j * np.pi * np.dot(transverse_k, r)) * matrix
    return device


def band_structure(
    h_00, h_01, h_10, s_blocks: tuple | None, num_k_points: int
) -> np.ndarray:
    """Computes the band structure of a periodic lead.

    Solves H(k) v = E S(k) v, falling back to the standard problem when
    no overlap is given. The k points are processed in chunks, since the
    full batch of (num_k_points, n, n) matrices does not fit in memory
    for large unit cells.

    """
    n = h_00.shape[-1]

    # h_k, s_k and the solver's temporaries are all (chunk, n, n)
    # complex128; allow for roughly eight such arrays at a time.
    chunk = max(1, K_CHUNK_BYTES // (n * n * 16 * 8))

    k = xp.linspace(-xp.pi, xp.pi, num_k_points)
    e_k = np.empty((num_k_points, n))

    for start in range(0, num_k_points, chunk):
        k_chunk = k[start : start + chunk]
        minus = xp.exp(-1j * k_chunk)[:, xp.newaxis, xp.newaxis]
        plus = xp.exp(1j * k_chunk)[:, xp.newaxis, xp.newaxis]

        h_k = h_01 * minus + h_00 + h_10 * plus
        s_k = None
        if s_blocks is not None:
            s_00, s_01, s_10 = s_blocks
            s_k = s_01 * minus + s_00 + s_10 * plus

        # NOTE: `eigvalsh` returns the eigenvalues in ascending order.
        w = eigvalsh(h_k, s_k, compute_module=xp.__name__, output_module="numpy")

        # The generalized solver factorizes S(k) with a Cholesky
        # decomposition, which silently yields NaN rather than raising if
        # S(k) is not positive definite. That happens when the blocks are
        # not the true lead blocks, i.e. the block size is wrong.
        if not np.all(np.isfinite(w)):
            raise ValueError(
                "eigensolve produced non-finite values; S(k) is not positive "
                "definite, which usually means the block size is wrong."
            )

        e_k[start : start + k_chunk.shape[0]] = w

        del h_k, s_k
        if xp.__name__ == "cupy":
            xp.get_default_memory_pool().free_all_blocks()

    return e_k


def report_blocksize(matrix: np.ndarray, tol: float = 1e-10) -> None:
    """Reports the smallest block size the matrix is tridiagonal for.

    A block-tridiagonal matrix with block size b has all its nonzeros
    within |i - j| < 2b, so the bandwidth fixes the smallest admissible
    block. Any multiple of it is also valid but folds the bands.

    """
    coo = matrix.tocoo()
    keep = np.abs(coo.data) > tol * np.abs(coo.data).max()
    bandwidth = int(np.abs(coo.row[keep] - coo.col[keep]).max())

    # A block size of b puts block (0, 2) at separations b + 1 and above,
    # so b >= bandwidth guarantees block tridiagonality. Bandwidth <= 2b
    # - 1 is only necessary, not sufficient: it is the largest separation
    # a block-tridiagonal matrix can have, but a matrix with that
    # bandwidth may still reach into block (0, 2).
    necessary = -(-(bandwidth + 1) // 2)

    n = matrix.shape[0]
    divisors = [b for b in range(necessary, n + 1) if n % b == 0]
    sufficient = [b for b in divisors if b >= bandwidth]
    print(
        f"  bandwidth = {bandwidth} -> blocksize >= {necessary} (necessary), "
        f">= {bandwidth} (sufficient)"
    )
    print(
        f"    divisors of {n}: {divisors[:5]}, "
        f"guaranteed: {sufficient[:3]}"
    )


def check_periodicity(
    matrix: sps.spmatrix, blocksize: int, label: str = "h", num_blocks: int = 5
) -> None:
    """Checks whether the device is block-Toeplitz at this block size.

    The contact band structure repeats the (0,0) and (0,1) blocks to
    build an infinite lead, which is only physical if every diagonal
    block is the same and every off-diagonal block is the same.

    """
    available = matrix.shape[0] // blocksize
    reference = block(matrix, blocksize, 0, 0)
    scale = np.abs(reference).max()

    print(f"  periodicity of '{label}' (max|{label}_ii - {label}_00|):")
    for i in range(1, min(num_blocks, available)):
        difference = np.abs(block(matrix, blocksize, i, i) - reference).max()
        print(f"    block {i:3d}: {difference:.3e}  ({difference / scale:7.2%})")


def find_period(matrix: sps.spmatrix, max_period: int = 8192) -> None:
    """Finds the repeat length of the on-site energies.

    The diagonal of a periodic region repeats with the period of the
    unit cell, which gives the contact block size independently of how
    the device as a whole is partitioned.

    """
    diagonal = np.asarray(matrix.diagonal()).real
    window = min(diagonal.size, 4 * max_period)
    d = diagonal[:window]
    scale = np.abs(d).max()

    periods = []
    for p in range(1, min(max_period, window // 2) + 1):
        if np.abs(d[p:] - d[:-p]).max() <= 1e-8 * scale:
            periods.append(p)
            if len(periods) == 3:
                break

    print(f"  on-site periods within the first {window} orbitals: {periods}")


def check_block_tridiagonal(
    matrix: sps.spmatrix, blocksize: int, label: str = "h", tol: float = 1e-10
) -> None:
    """Checks block tridiagonality over the whole matrix.

    Inspecting only block (0, 2) is not enough: the leading blocks of a
    device are the contact region and can be cleaner than the bulk.

    """
    coo = matrix.tocoo()
    keep = np.abs(coo.data) > tol * np.abs(coo.data).max()
    distance = np.abs(coo.row[keep] // blocksize - coo.col[keep] // blocksize)

    outside = distance > 1
    if not outside.any():
        print(f"  {label} is block-tridiagonal at blocksize={blocksize}")
        return

    print(
        f"  WARNING: {int(outside.sum())} entries of '{label}' lie outside the "
        f"block-tridiagonal structure, up to {int(distance.max())} blocks away, "
        f"max magnitude {np.abs(coo.data[keep][outside]).max():.3e}. The lead "
        f"blocks are not a valid periodic lead at this block size."
    )


def check_lead_blocks(matrix: sps.spmatrix, blocksize: int, label: str = "h") -> None:
    """Checks the assumptions the contact band structure relies on.

    The band structure only uses the (0,0), (0,1) and (1,0) blocks, which
    is exact only if the matrix is block-tridiagonal. Wannier and
    Gaussian bases have long-range tails, so check they were truncated.

    """
    m_01 = block(matrix, blocksize, 0, 1)
    m_01_norm = np.abs(m_01).max()
    m_02_norm = np.abs(block(matrix, blocksize, 0, 2)).max()
    print(f"  max|{label}_01| = {m_01_norm:.3e}, max|{label}_02| = {m_02_norm:.3e}")
    if m_02_norm > 1e-3 * m_01_norm:
        print(
            f"  WARNING: matrix is not block-tridiagonal at this block size. "
            f"The three-block band structure neglects the second-neighbor "
            f"coupling of '{label}'."
        )

    hermiticity = np.abs(block(matrix, blocksize, 1, 0) - m_01.conj().T).max()
    print(f"  hermiticity error = {hermiticity:.3e}")


def check_overlap(s_blocks: tuple) -> bool:
    """Reports the spectrum of S(k) at the edges of the Brillouin zone.

    The generalized eigensolve factorizes S(k) with a Cholesky
    decomposition, which requires S(k) to be positive definite. A
    non-positive smallest eigenvalue means the blocks are not a valid
    periodic lead; a tiny positive one means the basis is close to
    linearly dependent.

    """
    s_00, s_01, s_10 = s_blocks

    # A Gram matrix of normalized basis functions has a unit diagonal.
    diagonal = get_host(xp.diag(s_00).real)
    print(f"  diag(s_00): {diagonal.min():.4f} .. {diagonal.max():.4f}")

    # s_00 is a principal submatrix of the device overlap, so it must be
    # positive definite whatever the block structure is. If it is not,
    # the stored matrix is not an overlap, or the blocks do not group
    # contiguous orbitals.
    w_00 = eigvalsh(s_00, compute_module=xp.__name__, output_module="numpy")
    print(f"  s_00 eigenvalues: {w_00.min():.3e} .. {w_00.max():.3e}")
    if w_00.min() <= 0:
        print("  WARNING: s_00 itself is not positive definite.")

    positive_definite = True
    for name, phase in (("0", 1.0), ("pi", -1.0)):
        s_k = phase * s_01 + s_00 + phase * s_10
        w = eigvalsh(s_k, compute_module=xp.__name__, output_module="numpy")
        print(
            f"  S(k={name}) eigenvalues: {w.min():.3e} .. {w.max():.3e}"
            f"  (condition {w.max() / abs(w.min()):.2e})"
        )
        if w.min() <= 0:
            positive_definite = False
            print(
                f"  WARNING: S(k={name}) is not positive definite; the "
                f"Cholesky factorization will fail."
            )

        del s_k
        if xp.__name__ == "cupy":
            xp.get_default_memory_pool().free_all_blocks()

    return positive_definite


def scan_lead_offsets(
    matrix: sps.spmatrix, blocksize: int, num_offsets: int = 5
) -> None:
    """Reports where along the device the lead blocks are valid.

    A device can start on a partial cell or inside a contact region, so
    the leading blocks are not necessarily bulk-like. The blocks of a
    genuine periodic lead give a positive definite S(k).

    """
    num_blocks = matrix.shape[0] // blocksize
    offsets = sorted(
        set(np.linspace(0, num_blocks - 2, num_offsets).astype(int).tolist())
    )

    print("  scanning lead block offsets:")
    for offset in offsets:
        s_00, s_01, s_10 = lead_blocks(matrix, blocksize, offset)
        s_k = s_01 + s_00 + s_10
        w = eigvalsh(s_k, compute_module=xp.__name__, output_module="numpy")
        status = "ok" if w.min() > 0 else "NOT positive definite"
        print(f"    offset {offset:3d}: min eig S(k=0) = {w.min():11.3e}  {status}")

        del s_00, s_01, s_10, s_k
        if xp.__name__ == "cupy":
            xp.get_default_memory_pool().free_all_blocks()


def plot_hamiltonian(hamiltonian: sps.spmatrix, out_dir: Path) -> None:
    """Plots the magnitude of the whole Hamiltonian."""
    dense = hamiltonian.toarray()

    # A structural zero is not a small value and must not be drawn at the
    # bottom of the colour scale; it is masked and rendered as white
    # background, as in plotting/block-thomas/plot_lu_factors.py. The
    # previous form, log10(|H| + 1e-16), put every zero on the dark end of
    # viridis.
    with np.errstate(divide="ignore"):
        logmag = np.ma.masked_invalid(np.log10(np.abs(dense)))

    cmap = matplotlib.colormaps["viridis"].with_extremes(bad="white")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_facecolor("white")
    image = ax.matshow(logmag, cmap=cmap)
    fig.colorbar(image, ax=ax, label=r"$\log_{10}|H_{ij}|$")
    ax.set_title("Hamiltonian Matrix")
    fig.tight_layout()
    fig.savefig(out_dir / "hamiltonian_matrix.png", dpi=300)
    plt.close(fig)


def plot_band_structure(
    k: np.ndarray,
    e_k: np.ndarray,
    valence_band_edge: float,
    conduction_band_edge: float,
    out_dir: Path,
) -> None:
    """Plots the band structure, both full and zoomed on the gap."""
    band_gap = conduction_band_edge - valence_band_edge
    for name, ylim in (
        ("bandstructure.png", None),
        (
            "bandstructure_zoom.png",
            (valence_band_edge - band_gap, conduction_band_edge + band_gap),
        ),
    ):
        fig, ax = plt.subplots()
        ax.plot(k, e_k, color="k", lw=0.8)
        ax.axhline(valence_band_edge, color="tab:blue", ls="--", lw=0.8)
        ax.axhline(conduction_band_edge, color="tab:red", ls="--", lw=0.8)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel("Energy (eV)")
        ax.set_xlim(-np.pi, np.pi)
        if ylim is not None:
            ax.set_ylim(*ylim)
        fig.savefig(out_dir / name, dpi=300, bbox_inches="tight")
        plt.close(fig)


def report_gaps(e_k: np.ndarray, num_gaps: int = 5) -> None:
    """Reports the widest gaps in the spectrum.

    Use this to pick `mid_gap_energy`: a value that does not sit inside
    one of these gaps yields band edges that are really just adjacent
    levels within a band.

    """
    levels = np.sort(e_k.ravel())
    spacings = np.diff(levels)
    widest = np.argsort(spacings)[::-1][:num_gaps]
    print(f"  {num_gaps} widest gaps in the spectrum:")
    for i in widest:
        print(
            f"    {spacings[i]:8.4f} eV  between {levels[i]:9.4f} and "
            f"{levels[i + 1]:9.4f}  -> mid_gap_energy="
            f"{0.5 * (levels[i] + levels[i + 1]):.4f}"
        )


def widest_gaps(e_k: np.ndarray, num_gaps: int = 5) -> dict:
    """The `num_gaps` widest gaps in the spectrum, as report columns."""
    levels = np.sort(e_k.ravel())
    spacings = np.diff(levels)
    widest = np.argsort(spacings)[::-1][:num_gaps]
    return {
        "gap_eV": spacings[widest],
        "lower_eV": levels[widest],
        "upper_eV": levels[widest + 1],
        "mid_eV": 0.5 * (levels[widest] + levels[widest + 1]),
    }


def write_report(name: str, material: cli.Material, out_dir: Path,
                 k: np.ndarray, e_k: np.ndarray,
                 valence_band_edge: float, conduction_band_edge: float) -> None:
    """The computed band structure behind the two figures, as text beside
    them."""
    e_k = np.asarray(e_k)
    if e_k.size <= MAX_BAND_SAMPLES:
        bands = {"k": np.asarray(k, dtype=float)}
        for b in range(e_k.shape[1]):
            bands[f"band_{b:03d}_eV"] = e_k[:, b]
        series = {"band energies E(k), one column per band": bands}
        band_note = "full (k, band) grid tabulated"
    else:
        series = {"per-band envelope over k": {
            "band": np.arange(e_k.shape[1]),
            "min_eV": e_k.min(axis=0),
            "max_eV": e_k.max(axis=0),
            "mean_eV": e_k.mean(axis=0),
        }}
        band_note = (f"{e_k.size} samples exceed {MAX_BAND_SAMPLES}; the full "
                     f"E(k) grid is not tabulated, only its per-band envelope")
    series["widest gaps in the spectrum"] = widest_gaps(e_k)

    write_data_report(
        out_dir / "bandstructure_data.txt",
        title=f"contact band structure  —  {name}",
        source=str(material.inputs / "hamiltonian.mat"),
        config={
            "figures": "bandstructure.png, bandstructure_zoom.png, "
                       "hamiltonian_matrix.png",
            "block size": str(material.block_size),
            "transverse k": str(material.transverse_k),
            "k points": str(material.num_k_points),
            "lead offset": str(material.lead_offset),
            "mid-gap energy (eV)": str(material.mid_gap_energy),
            "valence band edge (eV)": f"{valence_band_edge:.6f}",
            "conduction band edge (eV)": f"{conduction_band_edge:.6f}",
            "band gap (eV)": f"{conduction_band_edge - valence_band_edge:.6f}",
            "eigenvalue range (eV)": f"{e_k.min():.6f} .. {e_k.max():.6f}",
        },
        series=series,
        notes=[f"Computed here, not read from a result file: the eigensolve "
               f"H(k) v = E S(k) v is this figure's subject.  {band_note}."],
    )


def run(name: str, material: cli.Material,
        hamiltonian_only: bool = False) -> None:
    """Computes and plots the band structure of a single material."""
    out_dir = PLOTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    blocksize = material.block_size

    # NOTE: An orthonormal basis (e.g. Wannier) needs no overlap matrix
    # and gives a standard eigenvalue problem. A non-orthogonal basis
    # (e.g. the CP2K Gaussian sets) ships an `overlap.mat` and requires
    # the generalized problem instead.
    hamiltonian = load_matrices(material.inputs / "hamiltonian.mat")
    overlap_path = material.inputs / "overlap.mat"
    overlap = load_matrices(overlap_path) if overlap_path.exists() else None

    print(
        f"  {len(hamiltonian)} neighbor cell(s), overlap={overlap is not None}, "
        f"transverse_k={material.transverse_k}"
    )
    if len(hamiltonian) > 1:
        report_hopping_grid(hamiltonian)

    # The matrices are already assembled along the transport direction;
    # the neighbor cells are the transverse periodic images.
    matrix = assemble_device(hamiltonian, material.transverse_k)
    print(f"  {matrix.shape=}, {blocksize=}")
    report_blocksize(matrix)

    if blocksize is None:
        print(f"  STOPPED: set block_size for '{name}' in "
              f"cli.MATERIALS from the report above.")
        return

    if matrix.shape[0] < 3 * blocksize:
        raise ValueError(
            f"Hamiltonian of shape {matrix.shape} is too small for "
            f"{blocksize=}; at least three blocks are needed."
        )

    check_lead_blocks(matrix, blocksize)
    check_block_tridiagonal(matrix, blocksize, label="h")
    check_periodicity(matrix, blocksize, label="h")
    find_period(matrix)
    plot_hamiltonian(matrix, out_dir)

    # The eigensolve below is the expensive part; skip it when only the
    # Hamiltonian figure is wanted.
    if hamiltonian_only:
        print(f"  wrote {out_dir / 'hamiltonian_matrix.png'}")
        return

    h_00, h_01, h_10 = lead_blocks(matrix, blocksize, material.lead_offset)

    s_blocks = None
    if overlap is not None:
        overlap_matrix = assemble_device(overlap, material.transverse_k)
        report_blocksize(overlap_matrix)
        check_lead_blocks(overlap_matrix, blocksize, label="s")
        check_block_tridiagonal(overlap_matrix, blocksize, label="s")
        s_blocks = lead_blocks(overlap_matrix, blocksize, material.lead_offset)

        if not check_overlap(s_blocks):
            scan_lead_offsets(overlap_matrix, blocksize)
            raise ValueError(
                "S(k) is not positive definite at "
                f"lead_offset={material.lead_offset}; pick an offset marked "
                "'ok' above."
            )
        del overlap_matrix, overlap

    num_k_points = material.num_k_points
    e_k = band_structure(h_00, h_01, h_10, s_blocks, num_k_points)
    k = np.linspace(-np.pi, np.pi, num_k_points)

    print(f"  eigenvalue range: {e_k.min():.4f} .. {e_k.max():.4f}")
    report_gaps(e_k)

    if material.mid_gap_energy is None:
        print(f"  STOPPED: set mid_gap_energy for '{name}' in "
              f"cli.MATERIALS from the report above.")
        return

    # `find_band_edges` takes a max/min over the two sides of the mid-gap
    # energy, which raises on an empty slice if it lies outside the
    # spectrum.
    if not (e_k.min() < material.mid_gap_energy < e_k.max()):
        raise ValueError(
            f"mid_gap_energy={material.mid_gap_energy} lies outside the "
            f"spectrum [{e_k.min():.4f}, {e_k.max():.4f}]."
        )

    valence_band_edge, conduction_band_edge = get_host(
        band_edges.find_band_edges(e_k, material.mid_gap_energy)
    )
    print(f"  valence band edge    = {valence_band_edge:.4f} eV")
    print(f"  conduction band edge = {conduction_band_edge:.4f} eV")
    print(f"  band gap             = {conduction_band_edge - valence_band_edge:.4f} eV")

    plot_band_structure(
        k, e_k, valence_band_edge, conduction_band_edge, out_dir
    )
    write_report(name, material, out_dir, k, e_k,
                 valence_band_edge, conduction_band_edge)
    print(f"  wrote plots to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "materials",
        nargs="*",
        default=list(MATERIALS),
        help="Materials to process. Defaults to all configured ones.",
    )
    parser.add_argument(
        "--hamiltonian-only",
        action="store_true",
        help="Only write hamiltonian_matrix.png, skipping the band structure.",
    )
    args = parser.parse_args()

    unknown = [name for name in args.materials if name not in MATERIALS]
    if unknown:
        parser.error(
            f"unknown material(s) {', '.join(unknown)}; "
            f"choose from {', '.join(MATERIALS)}"
        )

    failed = []
    for name in args.materials:
        material = MATERIALS[name]
        print(f"--- {name} ---")

        try:
            run(name, material, args.hamiltonian_only)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
