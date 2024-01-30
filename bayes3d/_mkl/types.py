# AUTOGENERATED! DO NOT EDIT! File to edit: ../../scripts/_mkl/notebooks/00a - Types.ipynb.

# %% auto 0
__all__ = [
    "Array",
    "Shape",
    "Bool",
    "Float",
    "Int",
    "FaceIndex",
    "FaceIndices",
    "ArrayN",
    "Array3",
    "Array2",
    "ArrayNx2",
    "ArrayNx3",
    "Matrix",
    "PrecisionMatrix",
    "CovarianceMatrix",
    "CholeskyMatrix",
    "SquareMatrix",
    "Vector",
    "Direction",
    "BaseVector",
]

# %% ../../scripts/_mkl/notebooks/00a - Types.ipynb 1
import jax
import jaxlib
import numpy as np

Array = np.ndarray | jax.Array
Shape = int | tuple[int, ...]
Bool = Array
Float = Array
Int = Array
FaceIndex = int
FaceIndices = Array
ArrayN = Array
Array3 = Array
Array2 = Array
ArrayNx2 = Array
ArrayNx3 = Array
Matrix = jaxlib.xla_extension.ArrayImpl
PrecisionMatrix = Matrix
CovarianceMatrix = Matrix
CholeskyMatrix = Matrix
SquareMatrix = Matrix
Vector = Array
Direction = Vector
BaseVector = Vector