"""Contact band structure of the CNT device (Wannier basis).

Extracts the periodic lead blocks (h_00, h_01, h_10) from the device
Hamiltonian and computes the contact band structure and band edges.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sps

from qttools import xp
from qttools.utils.gpu_utils import get_host
from quatrex.bandstructure import band_edges, contact
from quatrex.core.config import parse_config

simulation_dir = Path("/scratch/yimili/examples/w90/carbon-nanotube/qtbm")

config = parse_config(simulation_dir / "quatrex_config.toml")

# An energy that lies inside the band gap, used to separate the valence
# from the conduction bands.
MID_GAP_ENERGY = -3.8

NUM_K_POINTS = 401

# --- Load the device Hamiltonian -------------------------------------

# NOTE: The Wannier basis is orthonormal, so no overlap matrix is needed
# and the band structure follows from a standard eigenvalue problem.
hamiltonian = sps.load_npz(config.input_dir / "hamiltonian.npz").toarray()

try:
    block_sizes = np.load(config.input_dir / "block_sizes.npy")
    blocksize = int(block_sizes[0])
except FileNotFoundError:
    blocksize = 32

print(f"{hamiltonian.shape=}, {blocksize=}")

# --- Sanity checks on the lead blocks --------------------------------

# The contact band structure only uses h_00, h_01 and h_10, which is
# exact only if the Hamiltonian is block-tridiagonal. Wannier
# Hamiltonians have long-range tails, so check that they were truncated.
h_01_norm = np.abs(hamiltonian[:blocksize, blocksize : 2 * blocksize]).max()
h_02_norm = np.abs(hamiltonian[:blocksize, 2 * blocksize : 3 * blocksize]).max()
print(f"max|h_01| = {h_01_norm:.3e}, max|h_02| = {h_02_norm:.3e}")
if h_02_norm > 1e-3 * h_01_norm:
    print(
        "WARNING: Hamiltonian is not block-tridiagonal at this block size. "
        "The three-block band structure neglects the second-neighbor "
        "coupling and will be too flat away from Gamma."
    )

hermiticity = np.abs(
    hamiltonian[blocksize : 2 * blocksize, :blocksize]
    - hamiltonian[:blocksize, blocksize : 2 * blocksize].conj().T
).max()
print(f"hermiticity error = {hermiticity:.3e}")

# --- Plot the sparsity structure -------------------------------------

fig, ax = plt.subplots(figsize=(8, 6))
image = ax.matshow(
    np.log10(np.abs(hamiltonian) + 1e-16),
    cmap="viridis",
)
fig.colorbar(image, ax=ax, label=r"$\log_{10}|H_{ij}|$")
ax.set_title("Hamiltonian Matrix")
fig.tight_layout()
fig.savefig("hamiltonian_matrix.png", dpi=300)
plt.close(fig)

# --- Contact band structure ------------------------------------------

h_00 = xp.asarray(hamiltonian[:blocksize, :blocksize])
h_01 = xp.asarray(hamiltonian[:blocksize, blocksize : 2 * blocksize])
h_10 = xp.asarray(hamiltonian[blocksize : 2 * blocksize, :blocksize])

e_k = get_host(contact.contact_band_structure(h_10, h_00, h_01, NUM_K_POINTS))

k = np.linspace(-np.pi, np.pi, NUM_K_POINTS)

print(f"eigenvalue range: {e_k.min():.4f} .. {e_k.max():.4f}")

# --- Band edges ------------------------------------------------------

# `find_band_edges` takes a max/min over the two sides of the mid-gap
# energy, which raises on an empty slice if it lies outside the spectrum.
if not (e_k.min() < MID_GAP_ENERGY < e_k.max()):
    raise ValueError(
        f"{MID_GAP_ENERGY=} lies outside the spectrum "
        f"[{e_k.min():.4f}, {e_k.max():.4f}]."
    )

valence_band_edge, conduction_band_edge = get_host(
    band_edges.find_band_edges(e_k, MID_GAP_ENERGY)
)
band_gap = conduction_band_edge - valence_band_edge
print(f"valence band edge     = {valence_band_edge:.4f} eV")
print(f"conduction band edge  = {conduction_band_edge:.4f} eV")
print(f"band gap              = {band_gap:.4f} eV")

# --- Plot the band structure -----------------------------------------

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
    fig.savefig(name, dpi=300, bbox_inches="tight")
    plt.close(fig)
