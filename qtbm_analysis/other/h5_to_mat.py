from pathlib import Path
import glob
import scipy.io
from qttools.utils.hdf5_utils import load_hdf5_dict

SOURCE_DIR = Path("/scratch/yimili/examples/dev_1_CP2K/inputs")

# glob for all files with `.h5` extension in all subdirectories under SOURCE_DIR
h5_files = glob.glob(str(SOURCE_DIR / "**" / "*.h5"), recursive=True)
for file in h5_files:
    file = Path(file).resolve()
    print(f"Converting {file} ...")
    data = load_hdf5_dict(file)
    scipy.io.savemat(file.with_suffix(".mat"), data)
    print(f"  -> saved {file.with_suffix('.mat')}")