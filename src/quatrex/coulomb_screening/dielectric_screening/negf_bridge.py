from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io
from mpi4py.MPI import COMM_WORLD as global_comm

from qttools import sparse, xp
from qttools.comm import comm
from qttools.datastructures import DSDBSparse
from qttools.utils.gpu_utils import get_host
from quatrex.core.config import DeviceConfig, QuatrexConfig
from quatrex.core.statistics import bose_einstein
from quatrex.coulomb_screening.block_screening import (
    solve_environment_dressed_interaction,
)
from quatrex.device.inputs import (
    _create_matrix_from_unit_cells,
    load_matrix,
    trim_tight_binding_matrix,
)

from .equilibrium_screening import EquilibriumScreening
from .rpa_compute import (
    BrillouinZoneMesh,
    ScreeningChannels,
    build_uniform_brillouin_zone_mesh,
)


@dataclass(frozen=True)
class EquilibriumRPABridgeResult:
    """Cached real-space screened interactions for the NEGF Coulomb SSE path."""

    w_retarded_matrices: list[sparse.coo_matrix]
    w_lesser_matrices: list[sparse.coo_matrix]
    w_greater_matrices: list[sparse.coo_matrix]


def _advanced(matrix: np.ndarray) -> np.ndarray:
    return matrix.conj().T


def _compute_environment_greater(
    retarded: np.ndarray,
    lesser: np.ndarray,
) -> np.ndarray:
    return lesser + retarded - _advanced(retarded)


def _dress_environment_interactions(
    *,
    v_c: np.ndarray,
    v_ee: np.ndarray,
    v_ce: np.ndarray,
    v_ec: np.ndarray,
    p_ee_retarded_matrices: list[sparse.coo_matrix],
    p_ee_lesser_matrices: list[sparse.coo_matrix],
) -> EquilibriumRPABridgeResult:
    w_retarded_matrices: list[sparse.coo_matrix] = []
    w_lesser_matrices: list[sparse.coo_matrix] = []
    w_greater_matrices: list[sparse.coo_matrix] = []
    total_energies = len(p_ee_retarded_matrices)

    for energy_index, (p_retarded_matrix, p_lesser_matrix) in enumerate(
        zip(
            p_ee_retarded_matrices,
            p_ee_lesser_matrices,
            strict=True,
        ),
        start=1,
    ):
        if (
            total_energies <= 4
            or energy_index == 1
            or energy_index == total_energies
            or energy_index % max(1, total_energies // 4) == 0
        ):
            print(
                "Environment cache: dressing interaction "
                f"{energy_index}/{total_energies}...",
                flush=True,
            )
        dressed = solve_environment_dressed_interaction(
            v_c=v_c,
            v_ee=v_ee,
            v_ce=v_ce,
            v_ec=v_ec,
            p_ee_retarded=p_retarded_matrix.toarray(),
            p_ee_lesser=p_lesser_matrix.toarray(),
        )
        w_retarded_matrices.append(sparse.coo_matrix(dressed.retarded))
        w_lesser_matrices.append(sparse.coo_matrix(dressed.lesser))
        w_greater_matrices.append(
            sparse.coo_matrix(
                _compute_environment_greater(dressed.retarded, dressed.lesser)
            )
        )

    return EquilibriumRPABridgeResult(
        w_retarded_matrices=w_retarded_matrices,
        w_lesser_matrices=w_lesser_matrices,
        w_greater_matrices=w_greater_matrices,
    )


class EquilibriumRPAScreeningBridge:
    """Bridge equilibrium Bloch-RPA screening onto the NEGF screened-interaction buffers.

    The initial implementation is intentionally conservative:

    - it only supports device inputs constructed from a unit cell,
    - it requires no transverse k-point sampling,
    - it assumes the RPA periodic axis is the transport axis.

    These guards keep the new path isolated from the established NEGF workflow.
    """

    def __init__(
        self,
        config: QuatrexConfig,
        screening_energies: np.ndarray,
        template: DSDBSparse,
    ) -> None:
        self.config = config
        self.screening_energies = np.asarray(screening_energies, dtype=np.float64)
        self.template = template
        channels = ScreeningChannels(
            spin_degeneracy=config.coulomb_screening.spin_degeneracy,
            valley_degeneracy=config.coulomb_screening.valley_degeneracy,
        )
        self._solver = EquilibriumScreening(
            channels=channels,
            matrix_polarization=getattr(
                config.coulomb_screening,
                "matrix_valued_polarization",
                False,
            ),
            frequency_axis="real",
        )
        self._cached_result: EquilibriumRPABridgeResult | None = None
        self._cached_local_buffer_ids: tuple[int | None, int, int] | None = None

    def populate(
        self,
        w_retarded: DSDBSparse | None,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
    ) -> None:
        """Populate the distributed screened-interaction tensors used by SCBA."""

        if w_retarded is not None and w_retarded._data is None:
            w_retarded.allocate_data()
        if w_lesser._data is None:
            w_lesser.allocate_data()
        if w_greater._data is None:
            w_greater.allocate_data()

        if self._can_populate_screening_locally():
            local_buffer_ids = (
                id(w_retarded) if w_retarded is not None else None,
                id(w_lesser),
                id(w_greater),
            )
            if self._cached_local_buffer_ids == local_buffer_ids:
                if comm.rank == 0:
                    print(
                        "Environment cache: reused local environment screening "
                        "from in-memory buffers.",
                        flush=True,
                    )
                return

        if w_retarded is not None:
            w_retarded.data[:] = 0.0
        w_lesser.data[:] = 0.0
        w_greater.data[:] = 0.0

        if self._can_populate_saved_negf_screening_locally():
            self._populate_saved_negf_screening_locally(
                w_retarded=w_retarded,
                w_lesser=w_lesser,
                w_greater=w_greater,
            )
            return

        if self._can_populate_computed_rpa_screening_locally():
            self._populate_computed_rpa_screening_locally(
                w_retarded=w_retarded,
                w_lesser=w_lesser,
                w_greater=w_greater,
            )
            return

        result = self._cached_result or self._build_cached_result()
        self._cached_result = result

        local_energy_count = int(self.template.stack_section_sizes[self._stack_rank])
        energy_offset = int(
            np.sum(self.template.stack_section_sizes[: self._stack_rank])
        )

        for local_index in range(local_energy_count):
            global_index = energy_offset + local_index
            if w_retarded is not None:
                w_retarded.stack[(local_index,)] = result.w_retarded_matrices[
                    global_index
                ]
            w_lesser.stack[(local_index,)] = result.w_lesser_matrices[global_index]
            w_greater.stack[(local_index,)] = result.w_greater_matrices[global_index]

    def _can_populate_screening_locally(self) -> bool:
        return (
            self._can_populate_saved_negf_screening_locally()
            or self._can_populate_computed_rpa_screening_locally()
        )

    def _can_populate_saved_negf_screening_locally(self) -> bool:
        screening = self.config.environment_screening
        return screening.method == "negf" and screening.source == "file"

    def _can_populate_computed_rpa_screening_locally(self) -> bool:
        screening = self.config.environment_screening
        return screening.method == "rpa" and screening.source == "compute"

    def _populate_saved_negf_screening_locally(
        self,
        *,
        w_retarded: DSDBSparse | None,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
    ) -> None:
        """Populate environment-dressed interactions using only local energies.

        The original cache path builds all dressed interactions on rank 0 and
        broadcasts them to every rank. That is convenient for small grids, but it
        becomes the dominant memory bottleneck for high-resolution environment
        exports. Saved NEGF environment arrays are already energy-indexed on
        disk, so each stack rank can load and dress only its own energy section.
        """

        screening = self.config.environment_screening
        if screening.input_dir is None:
            raise ValueError(
                "environment.screening.source='file' requires environment.screening.input_dir."
            )

        local_energy_count = int(self.template.stack_section_sizes[self._stack_rank])
        energy_offset = int(
            np.sum(self.template.stack_section_sizes[: self._stack_rank])
        )
        energy_stop = energy_offset + local_energy_count

        if comm.rank == 0:
            print(
                "Environment cache: populating saved NEGF screening locally "
                "by MPI energy slice...",
                flush=True,
            )

        input_dir = screening.input_dir
        p_retarded = np.load(input_dir / "p_ee_retarded.npy", mmap_mode="r")
        p_lesser = np.load(input_dir / "p_ee_lesser.npy", mmap_mode="r")
        v_ee = np.asarray(np.load(input_dir / "v_ee.npy"), dtype=np.complex128)

        expected_energies = self.screening_energies.size
        if p_retarded.shape[0] != expected_energies:
            raise ValueError(
                "Saved environment retarded polarization has "
                f"{p_retarded.shape[0]} energies, expected {expected_energies}."
            )
        if p_lesser.shape != p_retarded.shape:
            raise ValueError(
                "Saved environment lesser polarization must have the same shape as the retarded polarization."
            )
        if p_retarded.ndim != 3:
            raise ValueError(
                "Saved environment polarization arrays must have shape (num_energies, n, n)."
            )
        if v_ee.ndim != 2 or v_ee.shape[0] != v_ee.shape[1]:
            raise ValueError(
                "Saved environment Coulomb matrix v_ee must be a square 2D array."
            )
        if p_retarded.shape[1:] != v_ee.shape:
            raise ValueError(
                "Saved environment polarization matrices must match the shape of v_ee."
            )

        v_c = self._load_bare_coulomb_matrix(
            self.config,
            matrix_name=self.config.coulomb_screening.coulomb_matrix_name,
        )
        v_ce, v_ec = self._load_environment_coupling_matrices(
            central_size=v_c.shape[0],
            environment_size=v_ee.shape[0],
        )

        for local_index, global_index in enumerate(range(energy_offset, energy_stop)):
            if (
                local_energy_count <= 4
                or local_index == 0
                or local_index == local_energy_count - 1
                or (local_index + 1) % max(1, local_energy_count // 4) == 0
            ):
                print(
                    "Environment cache: rank "
                    f"{comm.stack.rank} dressing local interaction "
                    f"{local_index + 1}/{local_energy_count} "
                    f"(global {global_index + 1}/{expected_energies})...",
                    flush=True,
                )

            dressed = solve_environment_dressed_interaction(
                v_c=v_c,
                v_ee=v_ee,
                v_ce=v_ce,
                v_ec=v_ec,
                p_ee_retarded=np.asarray(
                    p_retarded[global_index], dtype=np.complex128
                ),
                p_ee_lesser=np.asarray(p_lesser[global_index], dtype=np.complex128),
            )
            if w_retarded is not None:
                w_retarded.stack[(local_index,)] = sparse.coo_matrix(dressed.retarded)
            w_lesser.stack[(local_index,)] = sparse.coo_matrix(dressed.lesser)
            w_greater.stack[(local_index,)] = sparse.coo_matrix(
                _compute_environment_greater(dressed.retarded, dressed.lesser)
            )

        self._cached_local_buffer_ids = (
            id(w_retarded) if w_retarded is not None else None,
            id(w_lesser),
            id(w_greater),
        )

    def _populate_computed_rpa_screening_locally(
        self,
        *,
        w_retarded: DSDBSparse | None,
        w_lesser: DSDBSparse,
        w_greater: DSDBSparse,
    ) -> None:
        """Compute RPA environment screening only for this rank's energy slice.

        The root-cache path builds all RPA frequency points on rank 0 and then
        broadcasts full central-region W matrices. For matrix-valued RPA this is
        prohibitively expensive for large MoS2/hBN supercells. This path mirrors
        the saved-NEGF fast path: each MPI stack rank computes, transforms, and
        dresses only its local frequency section.
        """

        self._validate_supported_configuration()
        local_energy_count = int(self.template.stack_section_sizes[self._stack_rank])
        energy_offset = int(
            np.sum(self.template.stack_section_sizes[: self._stack_rank])
        )
        energy_stop = energy_offset + local_energy_count
        local_frequencies = self.screening_energies[energy_offset:energy_stop]

        if comm.rank == 0:
            print(
                "Environment cache: computing RPA screening locally "
                "by MPI energy slice...",
                flush=True,
            )

        screening = self.config.environment_screening
        environment_config = self._build_environment_config()
        mesh = self._build_mesh(frequencies=local_frequencies)
        chemical_potential = self._resolve_chemical_potential()

        print(
            "Environment cache: rank "
            f"{comm.stack.rank} loading environment RPA inputs "
            f"(nk={mesh.k_points.size}, nq={mesh.q_points.size}, "
            f"local nw={mesh.frequencies.size})...",
            flush=True,
        )
        inputs = self._solver.load_inputs_from_config(
            environment_config,
            hamiltonian_matrix_name=screening.hamiltonian_matrix_name,
            coulomb_matrix_name=screening.coulomb_matrix_name,
        )

        print(
            "Environment cache: rank "
            f"{comm.stack.rank} computing matrix-valued RPA P(q,w)...",
            flush=True,
        )
        polarization_result = self._solver.polarization_solver.solve_matrix_from_translation_blocks(
            translation_blocks=inputs.hamiltonian_blocks,
            mesh=mesh,
            chemical_potential=chemical_potential,
            temperature=self.config.coulomb_screening.temperature,
            periodic_axis=screening.periodic_axis,
            lattice_constant=screening.lattice_constant,
            broadening=screening.broadening,
        )
        p_retarded_qw = np.asarray(
            polarization_result.polarization,
            dtype=np.complex128,
        )
        if p_retarded_qw.ndim != 4:
            raise NotImplementedError(
                "RPA environment screening requires matrix_valued_polarization=True."
            )

        p_spectral_function = p_retarded_qw - np.swapaxes(
            p_retarded_qw.conj(), -1, -2
        )
        bose = np.asarray(
            bose_einstein(
                xp.asarray(local_frequencies),
                self.config.coulomb_screening.temperature,
            )
        )
        bose = bose.astype(np.complex128, copy=False)[
            np.newaxis, :, np.newaxis, np.newaxis
        ]
        p_lesser_qw = bose * p_spectral_function

        print(
            "Environment cache: rank "
            f"{comm.stack.rank} transforming local RPA P to transport matrices...",
            flush=True,
        )
        p_retarded_matrices = self._build_transport_matrices(
            environment_config,
            mesh,
            p_retarded_qw,
        )
        p_lesser_matrices = self._build_transport_matrices(
            environment_config,
            mesh,
            p_lesser_qw,
        )

        v_ee = self._load_bare_coulomb_matrix(
            environment_config,
            matrix_name=screening.coulomb_matrix_name,
        )
        v_c = self._load_bare_coulomb_matrix(
            self.config,
            matrix_name=self.config.coulomb_screening.coulomb_matrix_name,
        )
        v_ce, v_ec = self._load_environment_coupling_matrices(
            central_size=v_c.shape[0],
            environment_size=v_ee.shape[0],
        )

        expected_energies = self.screening_energies.size
        for local_index, global_index in enumerate(range(energy_offset, energy_stop)):
            if (
                local_energy_count <= 4
                or local_index == 0
                or local_index == local_energy_count - 1
                or (local_index + 1) % max(1, local_energy_count // 4) == 0
            ):
                print(
                    "Environment cache: rank "
                    f"{comm.stack.rank} dressing local RPA interaction "
                    f"{local_index + 1}/{local_energy_count} "
                    f"(global {global_index + 1}/{expected_energies})...",
                    flush=True,
                )

            dressed = solve_environment_dressed_interaction(
                v_c=v_c,
                v_ee=v_ee,
                v_ce=v_ce,
                v_ec=v_ec,
                p_ee_retarded=p_retarded_matrices[local_index].toarray(),
                p_ee_lesser=p_lesser_matrices[local_index].toarray(),
            )
            if w_retarded is not None:
                w_retarded.stack[(local_index,)] = sparse.coo_matrix(dressed.retarded)
            w_lesser.stack[(local_index,)] = sparse.coo_matrix(dressed.lesser)
            w_greater.stack[(local_index,)] = sparse.coo_matrix(
                _compute_environment_greater(dressed.retarded, dressed.lesser)
            )

        self._cached_local_buffer_ids = (
            id(w_retarded) if w_retarded is not None else None,
            id(w_lesser),
            id(w_greater),
        )

    @property
    def _stack_rank(self) -> int:
        return int(comm.stack.rank)

    def _build_cached_result(self) -> EquilibriumRPABridgeResult:
        result = None
        if global_comm.rank == 0:
            method = self.config.environment_screening.method or "rpa"
            print(
                f"Computing dielectric environment screening cache (method={method})...",
                flush=True,
            )
            result = self._build_cached_result_on_root()
            print("Finished dielectric environment screening cache.", flush=True)

        result = global_comm.bcast(result, root=0)
        return result

    def _build_cached_result_on_root(self) -> EquilibriumRPABridgeResult:
        screening = self.config.environment_screening
        if screening.source == "file":
            print(
                "Environment cache: loading saved environment screening export...",
                flush=True,
            )
            v_ee, p_retarded_matrices, p_lesser_matrices = (
                self._load_saved_environment_screening()
            )
        else:
            if screening.method == "negf":
                raise NotImplementedError(
                    "environment.screening.method='negf' currently supports source='file' only. "
                    "Run 'quatrex export-environment ...' first, then point environment.screening.input_dir to the saved export."
                )
            self._validate_supported_configuration()
            environment_config = self._build_environment_config()
            mesh = self._build_mesh()
            chemical_potential = self._resolve_chemical_potential()
            print(
                "Environment cache: loading Hamiltonian and Coulomb inputs "
                f"(nk={mesh.k_points.size}, nq={mesh.q_points.size}, "
                f"nw={mesh.frequencies.size})...",
                flush=True,
            )
            inputs = self._solver.load_inputs_from_config(
                environment_config,
                hamiltonian_matrix_name=screening.hamiltonian_matrix_name,
                coulomb_matrix_name=screening.coulomb_matrix_name,
            )
            scaled_inputs = type(inputs)(
                hamiltonian_blocks=inputs.hamiltonian_blocks,
                coulomb_blocks={
                    translation: block / self.config.coulomb_screening.epsilon_r
                    for translation, block in inputs.coulomb_blocks.items()
                },
            )

            print(
                "Environment cache: computing screened interactions on q/frequency grid...",
                flush=True,
            )
            grid_result = self._solver.solve_grid_from_inputs(
                inputs=scaled_inputs,
                mesh=mesh,
                chemical_potential=chemical_potential,
                temperature=self.config.coulomb_screening.temperature,
                periodic_axis=screening.periodic_axis,
                lattice_constant=screening.lattice_constant,
                broadening=screening.broadening,
            )

            print(
                "Environment cache: building lesser/greater spectral functions...",
                flush=True,
            )
            p_retarded_qw = np.asarray(
                grid_result.polarization_result.polarization,
                dtype=np.complex128,
            )
            if p_retarded_qw.ndim != 4:
                raise NotImplementedError(
                    "RPA environment screening requires matrix_valued_polarization=True."
                )
            p_spectral_function = p_retarded_qw - np.swapaxes(
                p_retarded_qw.conj(), -1, -2
            )
            bose = np.asarray(
                bose_einstein(
                    xp.asarray(self.screening_energies),
                    self.config.coulomb_screening.temperature,
                )
            )
            bose = bose.astype(np.complex128, copy=False)[
                np.newaxis, :, np.newaxis, np.newaxis
            ]
            p_lesser_qw = bose * p_spectral_function

            print(
                "Environment cache: transforming environment retarded P to transport matrices...",
                flush=True,
            )
            p_retarded_matrices = self._build_transport_matrices(
                environment_config,
                mesh,
                p_retarded_qw,
            )
            print(
                "Environment cache: transforming environment lesser P to transport matrices...",
                flush=True,
            )
            p_lesser_matrices = self._build_transport_matrices(
                environment_config,
                mesh,
                p_lesser_qw,
            )

            v_ee = self._load_bare_coulomb_matrix(
                environment_config,
                matrix_name=screening.coulomb_matrix_name,
            )

        print(
            "Environment cache: loading central bare Coulomb matrix v_c...",
            flush=True,
        )
        v_c = self._load_bare_coulomb_matrix(
            self.config,
            matrix_name=self.config.coulomb_screening.coulomb_matrix_name,
        )
        print(
            f"Environment cache: loaded v_c with shape {v_c.shape}.",
            flush=True,
        )
        print(
            "Environment cache: loading environment coupling matrices v_ce and v_ec...",
            flush=True,
        )
        v_ce, v_ec = self._load_environment_coupling_matrices(
            central_size=v_c.shape[0],
            environment_size=v_ee.shape[0],
        )
        print(
            "Environment cache: loaded coupling matrices with shapes "
            f"v_ce={v_ce.shape}, v_ec={v_ec.shape}.",
            flush=True,
        )

        print(
            "Environment cache: dressing the central Coulomb interaction with environment screening...",
            flush=True,
        )
        return _dress_environment_interactions(
            v_c=v_c,
            v_ee=v_ee,
            v_ce=v_ce,
            v_ec=v_ec,
            p_ee_retarded_matrices=p_retarded_matrices,
            p_ee_lesser_matrices=p_lesser_matrices,
        )

    def _load_saved_environment_screening(
        self,
    ) -> tuple[np.ndarray, list[sparse.coo_matrix], list[sparse.coo_matrix]]:
        screening = self.config.environment_screening
        if screening.input_dir is None:
            raise ValueError(
                "environment.screening.source='file' requires environment.screening.input_dir."
            )

        input_dir = screening.input_dir
        print(
            f"Environment cache: reading saved arrays from {input_dir}...",
            flush=True,
        )
        p_retarded = np.load(input_dir / "p_ee_retarded.npy")
        p_lesser = np.load(input_dir / "p_ee_lesser.npy")
        v_ee = np.asarray(np.load(input_dir / "v_ee.npy"), dtype=np.complex128)
        print(
            "Environment cache: loaded arrays with shapes "
            f"p_retarded={p_retarded.shape}, "
            f"p_lesser={p_lesser.shape}, "
            f"v_ee={v_ee.shape}.",
            flush=True,
        )

        expected_energies = self.screening_energies.size
        if p_retarded.shape[0] != expected_energies:
            raise ValueError(
                "Saved environment retarded polarization has "
                f"{p_retarded.shape[0]} energies, expected {expected_energies}."
            )
        if p_lesser.shape != p_retarded.shape:
            raise ValueError(
                "Saved environment lesser polarization must have the same shape as the retarded polarization."
            )
        if p_retarded.ndim != 3:
            raise ValueError(
                "Saved environment polarization arrays must have shape (num_energies, n, n)."
            )
        if v_ee.ndim != 2 or v_ee.shape[0] != v_ee.shape[1]:
            raise ValueError(
                "Saved environment Coulomb matrix v_ee must be a square 2D array."
            )
        if p_retarded.shape[1:] != v_ee.shape:
            raise ValueError(
                "Saved environment polarization matrices must match the shape of v_ee."
            )

        print(
            "Environment cache: converting saved polarization arrays to sparse matrices...",
            flush=True,
        )
        p_retarded_matrices = [
            sparse.coo_matrix(np.asarray(matrix, dtype=np.complex128))
            for matrix in p_retarded
        ]
        p_lesser_matrices = [
            sparse.coo_matrix(np.asarray(matrix, dtype=np.complex128))
            for matrix in p_lesser
        ]
        print(
            "Environment cache: finished sparse conversion for "
            f"{len(p_retarded_matrices)} energy points.",
            flush=True,
        )
        return v_ee, p_retarded_matrices, p_lesser_matrices

    def _build_mesh(self, frequencies: np.ndarray | None = None) -> BrillouinZoneMesh:
        screening = self.config.environment_screening
        base_mesh = build_uniform_brillouin_zone_mesh(
            num_k_points=screening.num_k_points,
            num_q_points=screening.num_q_points,
            num_frequencies=1,
            max_frequency=0.0,
            lattice_constant=screening.lattice_constant,
            include_zero_q=screening.include_zero_q,
        )
        return BrillouinZoneMesh(
            k_points=base_mesh.k_points,
            q_points=base_mesh.q_points,
            frequencies=(
                self.screening_energies
                if frequencies is None
                else np.asarray(frequencies, dtype=np.float64)
            ),
        )

    def _build_transport_matrices(
        self,
        config: QuatrexConfig,
        mesh: BrillouinZoneMesh,
        matrices_qw: np.ndarray,
    ) -> list[sparse.coo_matrix]:
        transport_axis = "xyz".index(config.device.transport_direction)
        periodic_axis = config.environment_screening.periodic_axis
        if periodic_axis is None:
            periodic_axis = transport_axis

        q_points = np.asarray(mesh.q_points, dtype=np.float64)
        nq = q_points.size
        max_translation = nq // 2
        translations = np.arange(-max_translation, max_translation + 1, dtype=int)
        axis_sizes = [1, 1, 1]
        axis_sizes[periodic_axis] = translations.size

        matrices: list[sparse.coo_matrix] = []
        for frequency_index in range(matrices_qw.shape[1]):
            q_slice = matrices_qw[:, frequency_index]
            unit_cells = np.zeros(
                tuple(axis_sizes) + q_slice.shape[-2:],
                dtype=np.complex128,
            )
            for translation in translations:
                block = (
                    np.sum(
                        np.exp(
                            -1j
                            * q_points
                            * translation
                            * config.environment_screening.lattice_constant
                        )[:, np.newaxis, np.newaxis]
                        * q_slice,
                        axis=0,
                        dtype=np.complex128,
                    )
                    / nq
                )
                block_index = [0, 0, 0]
                block_index[periodic_axis] = translation + max_translation
                unit_cells[tuple(block_index)] = block

            # The RPA bridge reconstructs translation blocks only along the
            # selected periodic/transport axis. Transverse directions are already
            # folded into the q-resolved response, so requesting their real-space
            # neighbor cutoffs here would require blocks that were never built.
            neighbor_cell_cutoff = np.zeros(3, dtype=int)
            requested_cutoff = np.asarray(config.device.neighbor_cell_cutoff, dtype=int)
            neighbor_cell_cutoff[periodic_axis] = requested_cutoff[periodic_axis]
            if max_translation < neighbor_cell_cutoff[periodic_axis]:
                raise ValueError(
                    "RPA q-grid contains too few translation blocks for the requested "
                    "transport-axis neighbor cutoff. "
                    f"Need at least {2 * neighbor_cell_cutoff[periodic_axis] + 1} "
                    f"q points, got {nq}."
                )
            trim_slices = [slice(None), slice(None), slice(None)]
            axis_cutoff = int(neighbor_cell_cutoff[periodic_axis])
            center = max_translation
            trim_slices[periodic_axis] = slice(
                center - axis_cutoff, center + axis_cutoff + 1
            )
            unit_cells = unit_cells[tuple(trim_slices)]
            matrix_sparray, __, __ = _create_matrix_from_unit_cells(
                config, unit_cells
            )
            matrices.append(matrix_sparray.astype(xp.complex128))

        return matrices

    def _build_environment_config(self) -> QuatrexConfig:
        if self.config.environment is None or not self.config.environment.enabled:
            raise NotImplementedError(
                "Dielectric environment screening requires an enabled [environment] section."
            )

        environment = self.config.environment
        device = DeviceConfig(
            construct_from_unit_cell=environment.construct_from_unit_cell,
            structure_file=environment.structure_file or "structure.xyz",
            neighbor_cell_cutoff=environment.neighbor_cell_cutoff,
            num_transport_cells=environment.num_transport_cells,
            transport_direction=environment.transport_direction,
            block_size=environment.block_size,
            kpoint_grid=environment.kpoint_grid,
            kpoint_shift=environment.kpoint_shift,
            orthogonal_basis=environment.orthogonal_basis,
            num_orbitals_per_atom=self.config.device.num_orbitals_per_atom,
        )
        electron_solver = self.config.electron.solver.model_copy(
            update={"compute_current": False}
        )
        electron = self.config.electron.model_copy(
            update={
                "solver": electron_solver,
                "left_fermi_level": self.config.electron.fermi_level,
                "right_fermi_level": self.config.electron.fermi_level,
                "flatband": False,
                "band_edge_tracking": None,
            }
        )
        screening = environment.screening
        scba = self.config.scba.model_copy(
            update={
                "coulomb_screening": True,
                "phonon": False,
                "photon": False,
            }
        )
        coulomb_screening = self.config.coulomb_screening.model_copy(
            update={
                "hamiltonian_matrix_name": screening.hamiltonian_matrix_name,
                "coulomb_matrix_name": screening.coulomb_matrix_name,
                "num_k_points": screening.num_k_points,
                "num_q_points": screening.num_q_points,
                "num_frequencies": screening.num_frequencies,
                "max_frequency": screening.max_frequency,
                "include_zero_q": screening.include_zero_q,
                "periodic_axis": screening.periodic_axis,
                "lattice_constant": screening.lattice_constant,
                "chemical_potential": screening.chemical_potential,
                "broadening": screening.broadening,
                "matrix_valued_polarization": screening.matrix_valued_polarization,
                "spin_degeneracy": screening.spin_degeneracy,
                "valley_degeneracy": screening.valley_degeneracy,
            }
        )
        return self.config.model_copy(
            update={
                "device": device,
                "electron": electron,
                "scba": scba,
                "coulomb_screening": coulomb_screening,
                "input_dir": environment.input_dir,
                "output_dir": self.config.output_dir / "environment",
                "environment": None,
            },
            deep=True,
        )

    def _load_bare_coulomb_matrix(
        self,
        config: QuatrexConfig,
        *,
        matrix_name: str,
    ) -> np.ndarray:
        cache_dir = config.output_dir / "environment"
        cache_path = cache_dir / f"{matrix_name}_epsilon_{config.coulomb_screening.epsilon_r:g}_dense.npy"
        if cache_path.is_file():
            print(
                f"Environment cache: loading dense Coulomb matrix from {cache_path}...",
                flush=True,
            )
            dense = np.load(cache_path)
            print(
                f"Environment cache: loaded cached dense Coulomb matrix with shape {dense.shape}.",
                flush=True,
            )
            return np.asarray(dense, dtype=np.complex128)

        print(
            f"Environment cache: load_matrix start for '{matrix_name}' from {config.input_dir}...",
            flush=True,
        )
        coulomb_matrix, __ = load_matrix(
            config=config,
            matrix_name=matrix_name,
            shift_kpoints=True,
        )
        print(
            "Environment cache: load_matrix finished; "
            f"symmetry={coulomb_matrix.symmetry}, distribution_state={coulomb_matrix.distribution_state}.",
            flush=True,
        )
        if not coulomb_matrix.symmetry:
            print("Environment cache: symmetrizing Coulomb matrix...", flush=True)
            coulomb_matrix.symmetrize()
            print("Environment cache: symmetrization finished.", flush=True)
        print(
            f"Environment cache: scaling Coulomb matrix by epsilon_r={config.coulomb_screening.epsilon_r}...",
            flush=True,
        )
        coulomb_matrix._data /= config.coulomb_screening.epsilon_r
        print("Environment cache: converting Coulomb matrix to dense...", flush=True)
        dense = np.asarray(get_host(coulomb_matrix.to_dense()[0]), dtype=np.complex128)
        print(
            f"Environment cache: dense conversion finished with shape {dense.shape}.",
            flush=True,
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Environment cache: saving dense Coulomb matrix to {cache_path}...", flush=True)
        np.save(cache_path, dense)
        return dense

    def _load_environment_coupling_matrices(
        self,
        *,
        central_size: int,
        environment_size: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        coupling = self.config.environment_coupling
        if coupling.mode == "zero":
            return (
                np.zeros((central_size, environment_size), dtype=np.complex128),
                np.zeros((environment_size, central_size), dtype=np.complex128),
            )
        v_ce = self._load_coupling_matrix(
            coupling.v_ce_file,
            shape=(central_size, environment_size),
        )
        v_ec = self._load_coupling_matrix(
            coupling.v_ec_file,
            shape=(environment_size, central_size),
        )
        if coupling.strength != 1.0:
            scale = np.sqrt(float(coupling.strength))
            v_ce = scale * v_ce
            v_ec = scale * v_ec
            print(
                "Environment cache: scaled coupling matrices by "
                f"sqrt(strength)={scale:g} (strength={coupling.strength:g}).",
                flush=True,
            )
        return v_ce, v_ec

    def _load_coupling_matrix(
        self,
        path: Path | None,
        *,
        shape: tuple[int, int],
    ) -> np.ndarray:
        if path is None:
            raise ValueError("Expected an environment coupling file path.")

        suffix = path.suffix.lower()
        if suffix == ".npy":
            matrix = np.load(path)
        elif suffix == ".npz":
            npz_file = np.load(path)
            if len(npz_file.files) != 1:
                raise ValueError(
                    f"Expected exactly one array in '{path}', found {len(npz_file.files)}."
                )
            matrix = npz_file[npz_file.files[0]]
        elif suffix == ".mat":
            mat_data = scipy.io.loadmat(path)
            keys = [key for key in mat_data if not key.startswith("__")]
            if len(keys) != 1:
                raise ValueError(
                    f"Expected exactly one matrix variable in '{path}', found {keys}."
                )
            matrix = mat_data[keys[0]]
        else:
            raise ValueError(
                f"Unsupported environment coupling file format '{path.suffix}'."
            )

        matrix = np.asarray(matrix, dtype=np.complex128)
        if matrix.shape != shape:
            raise ValueError(
                f"Expected environment coupling matrix '{path}' to have shape {shape}, got {matrix.shape}."
            )
        return matrix

    def _resolve_chemical_potential(self) -> float:
        if self.config.environment_screening.chemical_potential is not None:
            return float(self.config.environment_screening.chemical_potential)
        if self.config.electron.fermi_level is not None:
            return float(self.config.electron.fermi_level)
        return float(self.config.electron.left_fermi_level)

    def _validate_supported_configuration(self) -> None:
        if not self.config.device.construct_from_unit_cell:
            raise NotImplementedError(
                "RPA screening under NEGF currently supports only construct_from_unit_cell=True."
            )

        transport_axis = "xyz".index(self.config.device.transport_direction)
        periodic_axis = self.config.environment_screening.periodic_axis
        if periodic_axis is None:
            periodic_axis = transport_axis
        if periodic_axis != transport_axis:
            raise NotImplementedError(
                "RPA screening under NEGF currently requires the RPA periodic axis to match the transport direction."
            )

        transverse_k_grid = list(self.config.device.kpoint_grid)
        transverse_k_grid.pop(transport_axis)
        if any(k > 1 for k in transverse_k_grid):
            raise NotImplementedError(
                "RPA screening under NEGF currently supports only devices without transverse k-point sampling."
            )

        if not self.config.environment_screening.include_zero_q:
            raise NotImplementedError(
                "RPA screening under NEGF currently requires include_zero_q=True."
            )
