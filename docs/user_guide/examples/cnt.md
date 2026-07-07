# Carbon Nanotube

This example demonstrates how to obtain electronic structure information
as inputs for `quatrex` and how to perform different kinds of transport
simulations for a roughly 10 nm long (8, 0) single-wall carbon nanotube
(CNT).

{{ mol3d("../../assets/structures/carbon-nanotube.xyz", style={"stick":
{"radius": -1}, "sphere": {"scale": 0.25}}) }}

## Geometry and electronic structure

To generate the electronic structure of a CNT, the following steps are
required:

- Geometry generation with `ase`.
- Geometry relaxation with `vasp`.
- Self-consistent field (SCF) calculation with `vasp`.
- Wannierization with `wannier90` through `vasp`.

Each of these steps is described in detail below. Each step requires
specific input files from the previous step and should be run in the
order presented and in separate folders.

### Geometry

Create a new folder for the geometry generation and change into it:

```bash
mkdir geometry
cd geometry
```

The following Python script generates a (8, 0) armchair carbon nanotube
which results in a POSCAR structure file for VASP calculations. The
script uses the `ase` package to create the structure and add vacuum
space in the X and Y directions to prevent interactions across periodic
boundaries.

```Python
from ase.build import nanotube
from ase.io import write

# 1. Create a (8, 0) armchair carbon nanotube.
cnt = nanotube(n=8, m=0, length=1, bond=1.42, symbol='C')

# 2. Add vacuum space in the X and Y directions.
# This prevents the tube from interacting with itself across periodic boundaries.
# NOTE: 20 is a large vacuum value and could potentially be reduced.
cnt.center(vacuum=20.0, axis=(0, 1))

# 3. Save it as a VASP POSCAR file for DFT calculations.
write('POSCAR', cnt, format='vasp')
```

### Relaxation

Create a new folder for the relaxation, change into it, and copy the
POSCAR file from the geometry folder:

```bash
cd ..
mkdir relaxation
cd relaxation
cp ../geometry/POSCAR .
```

Create the `INCAR` file for the relaxation calculation:

```bash
cat << 'EOF' > INCAR
SYSTEM = CNT relaxation
KPAR = 11
ENCUT = 520
EDIFF = 1E-5
IBRION = 2
ISIF = 2
NSW = 200
ISMEAR = 0
SIGMA = 0.05
LWAVE = .FALSE.
LCHARG = .TRUE.
PREC = Accurate
EOF
```

Create the `KPOINTS` file for the relaxation calculation:

```bash
cat << 'EOF' > KPOINTS
KPOINTS for relaxation
0
Gamma
1 1 11
0 0 0
EOF
```

The `POTCAR` file for C and H is required and can be downloaded from the
VASP portal.

After preparing the input files, run the relaxation calculation with the
following command:

```bash
mpiexec -n {num_procs} vasp_std
```

This will result in a relaxed structure represented in the `CONTCAR`
file. The `CONTCAR` file can be used as input for the next step, which
is the electronic structure calculation.

!!! Warning
    This relaxation step only relaxes the atomic positions and does not
    relax the cell shape or volume. This would be done running the
    relaxation with different unit cell sizes and continuing with the
    energy minimal one.

### SCF

Create a new folder for the SCF calculation, change into it, and copy
the `CONTCAR`, `POTCAR`, and `KPOINTS` files from the relaxation folder:

```bash
cd ..
mkdir scf
cd scf
cp ../relaxation/CONTCAR POSCAR
cp ../relaxation/POTCAR .
cp ../relaxation/KPOINTS .
```

Create the `INCAR` file for the SCF calculation:

```bash
cat << 'EOF' > INCAR
SYSTEM = CNT static SCF
ENCUT = 520
# Parallelization over all k-points
# (adjust based on number of k-points and available cores)
KPAR = 11
EDIFF = 1E-6
ICHARG = 2
ISMEAR = 0
SIGMA = 0.05
LWAVE = .TRUE.
LCHARG = .TRUE.
PREC = Accurate
EOF
```

After preparing the input files, run the SCF calculation with the
following command:

```bash
mpiexec -n {num_procs} vasp_std
```

This will result in a converged electronic structure represented in the
`WAVECAR` and `CHGCAR` files.

### Wannierization

Create a new folder for the Wannierization, change into it, and copy the
`CONTCAR`, `POTCAR`, `KPOINTS`, and `WAVECAR` files from the relaxation and SCF folders:

```bash
cd ..
mkdir wannier
cd wannier
cp ../relaxation/CONTCAR POSCAR
cp ../relaxation/POTCAR .
cp ../relaxation/KPOINTS .
cp ../scf/WAVECAR .
cp ../scf/CHGCAR .
```

Create the `INCAR` file for the Wannierization calculation:

<details>
<summary>INCAR file creation</summary>

```bash
cat << 'EOF' > INCAR
SYSTEM = CNT wannier projection
ENCUT = 520
ALGO = None
NELM = 1
LWAVE = .FALSE.
LCHARG = .FALSE.
LWANNIER90 = .TRUE.
LWRITE_UNK = .TRUE.
NUM_WANN = 32
# NBANDS must be large enough to include the bands of interest
WANNIER90_WIN = "
dis_num_iter = 1000 # Number of disentanglement iterations
num_iter = 1000     # Number of Wannerisation iterations
guiding_centres = T # Not possible when an projection block is not specified

translate_home_cell = T
write_xyz = T

dis_win_min = -13 eV # disentangle window upper
dis_froz_min = -6 eV # -6 eV # frozen window min
dis_froz_max = -2 eV # -2 eV # frozen window max

bands_plot = .TRUE.
bands_plot_format=gnuplot
bands_num_points = 200
begin kpoint_path
   G 0.0 0.0 0 A 0.0 0.0 0.5
end kpoint_path

wannier_plot=.TRUE.
wannier_plot_format=xcrysden
wannier_plot_supercell = 1, 1, 7 
write_hr = T

# Following projections are for excluded bands
Begin Projections
    C:pz
End Projections
"
EOF
```
</details>

After preparing the input files, run the Wannierization calculation with
the following commands:

```bash
mpiexec -n {num_procs} vasp_std
mpirun -np 1 wannier90.x wannier90
```

This will result in a set of Wannier functions and a Hamiltonian in the
`wannier90_hr.dat` file, which first needs to be preprocessed before it
can be used as input for the transport simulations. This can be done
with the following Python script, which parses the contents of the
`wannier90_hr.dat` file and saves the Hamiltonian in an HDF5 file.

<details>
<summary>Preprocessing Python script</summary>

```Python
from pathlib import Path
import numpy as np

from qttools.utils.hdf5_utils import save_hdf5_dict

def transform_wannier_hr(
    path: Path, dtype = np.complex128
) -> None:
    """Parses the contents of a wannier90 `hr.dat` file.

    Parameters
    ----------
    path : Path
        Path to a `hr.dat` file.
    dtype : optional
        The data type for the Hamiltonian matrix elements. Defaults to
        `numpy.complex128`.

    """

    # Strip info from header.
    num_wannier_centers, num_hopping_cells = np.loadtxt(path, skiprows=1, max_rows=2, dtype=int)
    num_wannier_centers, num_hopping_cells = int(num_wannier_centers), int(num_hopping_cells)

    # Read wannier data (skipping degeneracy info). Wannier90 has some
    # Fortran formatting. Thus, the degeneracy information is arranged
    # into 15 values per line.
    degenerate_rows = int(np.ceil(num_hopping_cells / 15.0))
    wannier_data = np.loadtxt(path, skiprows=3 + degenerate_rows)

    cells = wannier_data[:, :3].astype(int)

    # Add for every unique cell an empty entry to the hamiltonian.
    hamiltonian = {
        f"[{cell[0]},{cell[1]},{cell[2]}]": np.zeros((num_wannier_centers, num_wannier_centers), dtype=dtype)
        for cell in np.unique(cells, axis=0)
    }

    # Obtain Hamiltonian elements.
    for line in wannier_data:
        cell = line[:3].astype(int)
        key = f"[{cell[0]},{cell[1]},{cell[2]}]"
        i, j = line[3:5].astype(int)
        hij_real, hij_imag = line[5:]
        hamiltonian[key][i - 1, j - 1] = hij_real + 1j * hij_imag

    save_hdf5_dict(
        filename="hamiltonian.h5",
        data=hamiltonian,
    )

transform_wannier_hr(
    path=Path("wannier90_hr.dat"),
    dtype=np.complex128,
)
```
</details>

!!! Note
    The preprocessing will be included in the CLI.


## Computing the transmission function

...

## Including phonon pseudo-scattering

...

## Band gap renormalization via the GW approximation

...
