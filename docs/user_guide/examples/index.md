# Example Simulation Setups

`quatrex` comes with a set of example simulation setups that
can be used to test the installation and to get familiar with the input
parameters. The examples are located in the `examples` directory of
`quatrex`. Each example contains a set of input files and a
`README.md` file that describes the simulation setup and how to run it.

The examples are also used as integration tests in the CI/CD pipeline.

The available example simulation setups are listed below. The examples
are grouped by the type of input files used to create the system
Hamiltonian and overlap matrix. The `cp2k` examples use [CP2K input
files](../input_data/#cp2k), while the `w90` examples use [Wannier90
input files](../input_data/#plane-wave-dft-wannier90).

```bash {title="Available Example Simulation Setups"}
quatrex/examples/
├── cp2k
│   ├── carbon-chain  # A simple chain of carbon atoms
│   │   ├── inputs
│   │   ├── phonon
│   │   ├── qtbm
│   │   └── qtbm-low-rank
│   └── graphene  # A layer of graphene
│       ├── inputs
│       ├── qtbm
│       └── qtbm-low-rank
└── w90
    ├── carbon-nanotube  # An (8,0) carbon nanotube
    │   ├── gw
    │   ├── gw-dist
    │   ├── gw-unit-cell
    │   ├── inputs
    │   └── qtbm
    ├── mos2  # A monolayer of MoS2
    │   ├── gw-kpoints
    │   ├── gw-kpoints-symmetric
    │   └── inputs
    └── si-bulk  # Bulk crystalline silicon
        ├── inputs
        ├── qtbm
        └── qtbm-low-rank
```
