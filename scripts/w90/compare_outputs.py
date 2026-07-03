import numpy as np
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    for parent in (start, *start.parents):
        if (parent / "src" / "quatrex").is_dir():
            return parent
    raise RuntimeError("Could not find repository root containing src/quatrex.")


W90_ROOT = _find_repo_root(Path(__file__).resolve()) / "examples" / "w90"
ROOT = W90_ROOT / "carbon-nanotube" / "gw-unit-cell"


pairs = [("p_retarded_diagonal_1.npy", "polarization"), ("electron_ldos_1.npy", "ldos"), ("electron_density_1.npy", "electron density")]
for filename, label in pairs:
    a = np.load(ROOT / "outputs" / filename)
    b = np.load(ROOT / "outputs_negf_environment" / filename)
    diff = np.max(np.abs(a - b))
    norm = np.max(np.abs(a))
    rel = diff / norm if norm != 0 else diff
    print(label, "max abs diff =", diff, "relative =", rel)
