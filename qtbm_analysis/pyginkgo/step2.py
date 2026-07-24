import pyGinkgo as pg

dev = pg.device("cuda")
A = pg.read(device=dev, path="test_matrix.mtx", dtype="half", format="Csr")
print("read ok, shape:", A.shape)