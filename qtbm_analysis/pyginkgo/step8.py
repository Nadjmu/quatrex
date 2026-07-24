import pyGinkgo as pg
import numpy as np

dev = pg.device("cuda")
A = pg.read(device=dev, path="test_matrix.mtx", dtype="half", format="Csr")
n_rows = A.shape[0]
print("type of A:", type(A))

# LU factorization in half precision
L = pg.factor(A, kind="Lower", device=dev)
U = pg.factor(A, kind="Upper", device=dev)

print("L type:", type(L))
print("U type:", type(U))

b = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="half", fill=1.0)
x = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="half", fill=0.0)

# Solve Ly = b (lower triangular solve)
y = pg.as_tensor(device=dev, dim=(n_rows, 1), dtype="half", fill=0.0)
_, y = pg.triangular_solve(L, b, y, solver_args={"type": "Lower"})

# Solve Ux = y (upper triangular solve)
_, x = pg.triangular_solve(U, y, x, solver_args={"type": "Upper"})

print("solved")
print(np.array(x))