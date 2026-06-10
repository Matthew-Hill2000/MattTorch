target_sources(
  mattTorch
  PRIVATE kernels/cpu/tensor-tensor/elementwise/elementwise.cpp
          kernels/cpu/tensor-tensor/matrixmult/matrix.cpp
          kernels/cpu/tensor-tensor/matrixmult/matrixBlockTransposeVector.cpp
          kernels/cpu/tensor-tensor/matrixmult/matrixBlockPackedVector.cpp
          kernels/cpu/tensor-tensor/matrixmult/transposeMultLHS.cpp
          kernels/cpu/tensor-tensor/matrixmult/transposeMultRHS.cpp
          kernels/cpu/tensor-scalar/tensorScalarElementwise.cpp
          kernels/cpu/tensor/tensorElementwise.cpp
          tensorStorage/tensorStorage.cpp
          tensor/tensor.cpp
          tensor/tensorAccesssors.cpp
          tensor/tensorBackProp.cpp
          tensor/tensorOverloads.cpp)
