import pyGinkgo as pg
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

n = 100
A_dense = np.eye(n) * 4.0
A_dense += np.diag(-1.0 * np.ones(n - 1), 1)
A_dense += np.diag(-1.0 * np.ones(n - 1), -1)
A_csr = sp.csr_matrix(A_dense)
sio.mmwrite("test_matrix.mtx", A_csr)
print("mtx written ok")