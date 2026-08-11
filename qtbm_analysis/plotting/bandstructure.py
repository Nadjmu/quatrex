"""Contact band structures of the example devices.

For each material the periodic lead blocks (h_00, h_01, h_10) are
extracted from the device Hamiltonian and used to compute the contact
band structure and the band edges.

Usage:
    python bandstructure.py               # all configured materials
    python bandstructure.py mos2 si-bulk  # a subset
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.io

from qttools import xp
from qttools.kernels.linalg import eigvalsh
from qttools.utils.gpu_utils import get_host
from quatrex.bandstructure import band_edges

EXAMPLES_DIR = Path("/scratch/yimili/examples")
PLOTS_DIR = Path("/scratch/yimili/plots")

NUM_K_POINTS = 401

# Memory budget for one chunk of k points. Lower this if the eigensolve
# runs out of device memory.
K_CHUNK_BYTES = 8 * 1024**3


@dataclass
class Material:
    """A device to compute the band structure of.

    Attributes
    ----------
    input_dir : Path
        The directory holding `hamiltonian.mat`.
    blocksize : int
        The size of one periodic lead block along the transport
        direction.
    mid_gap_energy : float
        An energy inside the band gap, used to separate the valence from
        the conduction bands.
    transverse_k : tuple
        The transverse momentum at which to evaluate the device, in
        fractional coordinates. The origin is Gamma.

    """

    input_dir: Path
    blocksize: int | None
    mid_gap_energy: float | None
    transverse_k: tuple[float, float, float] = (0.0, 0.0, 0.0)
    num_k_points: int = NUM_K_POINTS


# NOTE: Fill in `blocksize` and `mid_gap_energy` per material. Materials
# still set to `None` are skipped.
MATERIALS = {
    "carbon-nanotube": Material(
        input_dir=EXAMPLES_DIR / "w90/carbon-nanotube/inputs",
        blocksize=32,
        mid_gap_energy=-3.8,
    ),
    "si-bulk": Material(
        # 960 Si atoms * 4 orbitals / 30 cells along x = 128.
        input_dir=EXAMPLES_DIR / "w90/si-bulk/inputs",
        blocksize=128,
        mid_gap_energy=6.2,
    ),
    "carbon-chain": Material(
        input_dir=EXAMPLES_DIR / "cp2k/carbon-chain/inputs",
        blocksize=104,
        mid_gap_energy=-12,
    ),
    "graphene": Material(
        # 160 C atoms * 13 orbitals / 20 cells along x = 104.
        input_dir=EXAMPLES_DIR / "graphene/inputs",
        blocksize=104,
        mid_gap_energy=0.5,
    ),
}


def load_matrices(path: Path) -> dict[str, np.ndarray]:
    """Loads all matrices from a MATLAB file, keyed by neighbor cell."""
    mat = scipy.io.loadmat(path)
    return {k: v.toarray() for k, v in mat.items() if not k.startswith("__")}


def lead_blocks(matrix: np.ndarray, blocksize: int) -> tuple:
    """Extracts the (0,0), (0,1) and (1,0) blocks of a device matrix."""
    return (
        xp.asarray(matrix[:blocksize, :blocksize]),
        xp.asarray(matrix[:blocksize, blocksize : 2 * blocksize]),
        xp.asarray(matrix[blocksize : 2 * blocksize, :blocksize]),
    )


def report_hopping_grid(matrices: dict[str, np.ndarray]) -> None:
    """Reports the magnitude of each neighbor-cell matrix.

    The direction along which the couplings are strongest is a good
    indication of the transport direction.

    """
    keys = sorted(matrices, key=lambda k: tuple(int(i) for i in k.strip("[]").split(",")))
    print("  neighbor cells:")
    for key in keys:
        print(f"    {key:>14}  max|H| = {np.abs(matrices[key]).max():.4e}")


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
        e_k[start : start + k_chunk.shape[0]] = eigvalsh(
            h_k, s_k, compute_module=xp.__name__, output_module="numpy"
        )

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
    rows, cols = np.nonzero(np.abs(matrix) > tol * np.abs(matrix).max())
    bandwidth = int(np.abs(rows - cols).max())
    minimal = -(-(bandwidth + 1) // 2)

    n = matrix.shape[0]
    divisors = [b for b in range(minimal, n + 1) if n % b == 0]
    print(
        f"  bandwidth = {bandwidth} -> smallest blocksize = {minimal}"
        f", admissible divisors of {n}: {divisors[:5]}"
    )


def check_lead_blocks(hamiltonian: np.ndarray, blocksize: int) -> None:
    """Checks the assumptions the contact band structure relies on.

    The band structure only uses h_00, h_01 and h_10, which is exact
    only if the Hamiltonian is block-tridiagonal. Wannier Hamiltonians
    have long-range tails, so check that they were truncated.

    """
    h_01_norm = np.abs(hamiltonian[:blocksize, blocksize : 2 * blocksize]).max()
    h_02_norm = np.abs(hamiltonian[:blocksize, 2 * blocksize : 3 * blocksize]).max()
    print(f"  max|h_01| = {h_01_norm:.3e}, max|h_02| = {h_02_norm:.3e}")
    if h_02_norm > 1e-3 * h_01_norm:
        print(
            "  WARNING: Hamiltonian is not block-tridiagonal at this block "
            "size. The three-block band structure neglects the "
            "second-neighbor coupling and will be too flat away from Gamma."
        )

    hermiticity = np.abs(
        hamiltonian[blocksize : 2 * blocksize, :blocksize]
        - hamiltonian[:blocksize, blocksize : 2 * blocksize].conj().T
    ).max()
    print(f"  hermiticity error = {hermiticity:.3e}")


def plot_hamiltonian(hamiltonian: np.ndarray, out_dir: Path) -> None:
    """Plots the sparsity structure of the device Hamiltonian."""
    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.matshow(np.log10(np.abs(hamiltonian) + 1e-16), cmap="viridis")
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


def run(name: str, material: Material) -> None:
    """Computes and plots the band structure of a single material."""
    out_dir = PLOTS_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)

    blocksize = material.blocksize

    # NOTE: An orthonormal basis (e.g. Wannier) needs no overlap matrix
    # and gives a standard eigenvalue problem. A non-orthogonal basis
    # (e.g. the CP2K Gaussian sets) ships an `overlap.mat` and requires
    # the generalized problem instead.
    hamiltonian = load_matrices(material.input_dir / "hamiltonian.mat")
    overlap_path = material.input_dir / "overlap.mat"
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

    if matrix.shape[0] < 3 * blocksize:
        raise ValueError(
            f"Hamiltonian of shape {matrix.shape} is too small for "
            f"{blocksize=}; at least three blocks are needed."
        )

    check_lead_blocks(matrix, blocksize)
    plot_hamiltonian(matrix, out_dir)

    h_00, h_01, h_10 = lead_blocks(matrix, blocksize)
    s_blocks = (
        lead_blocks(assemble_device(overlap, material.transverse_k), blocksize)
        if overlap is not None
        else None
    )

    num_k_points = material.num_k_points
    e_k = band_structure(h_00, h_01, h_10, s_blocks, num_k_points)
    k = np.linspace(-np.pi, np.pi, num_k_points)

    print(f"  eigenvalue range: {e_k.min():.4f} .. {e_k.max():.4f}")
    report_gaps(e_k)

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
    print(f"  wrote plots to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "materials",
        nargs="*",
        default=list(MATERIALS),
        help="Materials to process. Defaults to all configured ones.",
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

        if material.mid_gap_energy is None:
            print("  SKIPPED: mid_gap_energy is not set.")
            continue

        try:
            run(name, material)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")
            failed.append(name)

    if failed:
        raise SystemExit(f"Failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()
