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
from qttools.utils.gpu_utils import get_host
from quatrex.bandstructure import band_edges, contact

EXAMPLES_DIR = Path("/scratch/yimili/examples")
PLOTS_DIR = Path("/scratch/yimili/plots")

NUM_K_POINTS = 401


@dataclass
class Material:
    """A device to compute the band structure of.

    Attributes
    ----------
    input_dir : Path
        The directory holding `hamiltonian.mat`.
    blocksize : int
        The size of one periodic lead block.
    mid_gap_energy : float
        An energy inside the band gap, used to separate the valence from
        the conduction bands.

    """

    input_dir: Path
    blocksize: int | None
    mid_gap_energy: float | None


# NOTE: Fill in `blocksize` and `mid_gap_energy` per material. Materials
# still set to `None` are skipped.
MATERIALS = {
    "carbon-nanotube": Material(
        input_dir=EXAMPLES_DIR / "w90/carbon-nanotube/inputs",
        blocksize=32,
        mid_gap_energy=-3.8,
    ),
    "si-bulk": Material(
        input_dir=EXAMPLES_DIR / "w90/si-bulk/inputs",
        blocksize=256,
        mid_gap_energy=6.2,
    ),
    "carbon-chain": Material(
        input_dir=EXAMPLES_DIR / "cp2k/carbon-chain/inputs",
        blocksize=104,
        mid_gap_energy=-12,
    ),
    "graphene": Material(
        input_dir=EXAMPLES_DIR / "graphene/inputs",
        blocksize=416,
        mid_gap_energy=0.5,
    ),
}


def load_hamiltonian(input_dir: Path) -> np.ndarray:
    """Loads the dense device Hamiltonian from `hamiltonian.mat`."""
    mat = scipy.io.loadmat(input_dir / "hamiltonian.mat")
    (key,) = [k for k in mat if not k.startswith("__")]
    return mat[key].toarray()


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

    # NOTE: The Wannier basis is orthonormal, so no overlap matrix is
    # needed and the band structure follows from a standard eigenvalue
    # problem.
    hamiltonian = load_hamiltonian(material.input_dir)
    print(f"  {hamiltonian.shape=}, {blocksize=}")

    if hamiltonian.shape[0] < 3 * blocksize:
        raise ValueError(
            f"Hamiltonian of shape {hamiltonian.shape} is too small for "
            f"{blocksize=}; at least three blocks are needed."
        )

    check_lead_blocks(hamiltonian, blocksize)
    plot_hamiltonian(hamiltonian, out_dir)

    h_00 = xp.asarray(hamiltonian[:blocksize, :blocksize])
    h_01 = xp.asarray(hamiltonian[:blocksize, blocksize : 2 * blocksize])
    h_10 = xp.asarray(hamiltonian[blocksize : 2 * blocksize, :blocksize])

    e_k = get_host(contact.contact_band_structure(h_10, h_00, h_01, NUM_K_POINTS))
    k = np.linspace(-np.pi, np.pi, NUM_K_POINTS)

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

        if material.blocksize is None or material.mid_gap_energy is None:
            print("  SKIPPED: blocksize and mid_gap_energy are not set.")
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
