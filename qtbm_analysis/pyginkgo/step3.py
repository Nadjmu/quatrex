import pyGinkgo as pg

dev = pg.device("cuda")
A = pg.read(device=dev, path="test_matrix.mtx", dtype="half", format="Csr")
n_rows = A.shape[0]

b = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="half", fill=1.0)
print("b created ok")

x = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="half", fill=0.0)
print("x created ok")