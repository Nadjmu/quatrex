import cupy
import pyGinkgo.pyGinkgoBindings as pGB

executor = pGB.CudaExecutor()

# CuPy → Ginkgo (zero-copy view via __cuda_array_interface__)
cp_arr = cupy.array([1.0, 2.0, 3.0], dtype=cupy.float64)
gko_arr = pGB.base.array_double(executor, cp_arr)

cp_mat = cupy.array([[1, 2], [3, 4]], dtype=cupy.float64)
gko_dense = pGB.matrix.dense_double(executor, cp_mat)

# Ginkgo → CuPy (zero-copy view via __cuda_array_interface__)
result = cupy.asarray(gko_arr)

print("here is the result:", result)

print([n for n in dir(pGB) if not n.startswith('_')])
print([n for n in dir(pGB.base) if not n.startswith('_')] if hasattr(pGB, 'base') else 'no base')
