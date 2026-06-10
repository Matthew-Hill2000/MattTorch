#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>

#include <cmath>
#include <functional>
#include <vector>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

std::vector<double> naiveMatMul(mattTorch::Tensor& a, mattTorch::Tensor& b,
                                int I, int K, int J) {
  std::vector<double> c(I * J, 0.0);
  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++) {
      double acc = 0;
      for (int k = 0; k < K; k++)
        acc += a.getValueDirect(i * K + k) * b.getValueDirect(k * J + j);
      c[i * J + j] = acc;
    }
  return c;
}

bool checkMatMul(const std::function<void(const double*, const double*, double*,
                                          int, int, int)>& kernel,
                 int I = 70, int K = 130, int J = 90) {
  mattTorch::Tensor a = randomTensor({I, K}, false);
  mattTorch::Tensor b = randomTensor({K, J}, false);
  double* r = createResultBuffer(I * J);
  kernel(a.getData(), b.getData(), r, I, K, J);
  auto ref = naiveMatMul(a, b, I, K, J);
  for (int i = 0; i < I * J; i++)
    if (std::abs(r[i] - ref[i]) > TOL) return false;
  return true;
}

bool testTransposeLHS() {
  const int I = 70, J = 90, K = 50;
  mattTorch::Tensor a = randomTensor({I, J}, false);
  mattTorch::Tensor b = randomTensor({I, K}, false);
  double* r = createResultBuffer(J * K);
  mattTorch::tensor::kernels::cpu::transposeMultBlockVectorLHS(
      a.getData(), b.getData(), r, I, J, K);
  for (int j = 0; j < J; j++)
    for (int k = 0; k < K; k++) {
      double acc = 0;
      for (int i = 0; i < I; i++)
        acc += a.getValueDirect(i * J + j) * b.getValueDirect(i * K + k);
      if (std::abs(r[j * K + k] - acc) > TOL) return false;
    }
  return true;
}

bool testTransposeRHS() {
  const int I = 70, J = 90, K = 50;
  mattTorch::Tensor a = randomTensor({I, J}, false);
  mattTorch::Tensor b = randomTensor({K, J}, false);
  double* r = createResultBuffer(I * K);
  mattTorch::tensor::kernels::cpu::transposeMultBlockVectorRHS(
      a.getData(), b.getData(), r, I, J, K);
  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++) {
      double acc = 0;
      for (int j = 0; j < J; j++)
        acc += a.getValueDirect(i * J + j) * b.getValueDirect(k * J + j);
      if (std::abs(r[i * K + k] - acc) > TOL) return false;
    }
  return true;
}

int main() {
  run("MatMul (naive)",
      checkMatMul(mattTorch::tensor::kernels::cpu::matrixMult));
  run("MatMul (transpose)",
      checkMatMul(mattTorch::tensor::kernels::cpu::matrixMultTranspose));
  run("MatMul (transpose+SIMD)",
      checkMatMul(mattTorch::tensor::kernels::cpu::matrixMultTransposeVector));
  run("MatMul (blocked)",
      checkMatMul(mattTorch::tensor::kernels::cpu::matrixMultBlockTranspose));
  run("MatMul (blocked+SIMD+OpenMP)",
      checkMatMul(
          mattTorch::tensor::kernels::cpu::matrixMultBlockTransposeVector));
  run("MatMul (MegaKernel)",
      checkMatMul(
          mattTorch::tensor::kernels::cpu::matrixMultBlockPackedVector));
  run("TransposeMul LHS (A^T @ B)", testTransposeLHS());
  run("TransposeMul RHS (A @ B^T)", testTransposeRHS());
}
