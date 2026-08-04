from pathlib import Path
from quatrex.core.config import parse_config, setup_context
from quatrex.device import Device
import numpy as np

config = parse_config(Path("/scratch/yimili/examples/dev_1_CP2K/qtbm/quatrex_config.toml"))
setup_context(config)

device = Device.__new__(Device)  # bypass __init__ to avoid the crash
device.config = config
device._init_hamiltonian()
device._init_lattice()
device._init_orbitals()

for i, c in enumerate(config.device.contacts):
    print(i, c.name, "origin:", c.origin)

right_cfg = config.device.contacts[1]  # confirm this is really "right" from the print above
origin = np.array(right_cfg.origin)
lattice_vectors = np.array(right_cfg.lattice_vectors)

rel = device.atom_coordinates - origin
frac = rel @ np.linalg.inv(lattice_vectors)
inside = np.nonzero(
    (frac[:,0]>=0)&(frac[:,0]<=1)&(frac[:,1]>=0)&(frac[:,1]<=1)&(frac[:,2]>=0)&(frac[:,2]<=1)
)[0]
print("atoms in right origin cell:", len(inside))
print("x-range of these atoms:", device.atom_coordinates[inside,0].min() if len(inside) else None,
      device.atom_coordinates[inside,0].max() if len(inside) else None)
print("x-range of ALL atoms:", device.atom_coordinates[:,0].min(), device.atom_coordinates[:,0].max())

# Now check the actual Hamiltonian coupling from these orbitals to the rest of the device
orbital_offsets = device.orbital_offsets
starts = orbital_offsets[inside]
ends = orbital_offsets[inside + 1]
orb_idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
H00 = device.hamiltonians[(0,0,0)]
all_orbitals = np.arange(H00.shape[0])
other_orbitals = np.setdiff1d(all_orbitals, orb_idx)
coupling_nnz = H00[orb_idx, :][:, other_orbitals].nnz
print("nonzero H coupling from right origin cell to rest of device:", coupling_nnz)