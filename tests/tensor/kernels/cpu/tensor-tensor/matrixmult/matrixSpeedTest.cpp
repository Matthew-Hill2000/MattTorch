#include <cblas.h>
#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchNaive(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int I,
                int K, int J) {
  mattTorch::tensor::kernels::cpu::matrixMult(a.getData(), b.getData(), r, I, K, J);
}
void benchTranspose(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int I,
                    int K, int J) {
  mattTorch::tensor::kernels::cpu::matrixMultTranspose(a.getData(), b.getData(), r, I, K, J);
}
void benchTransposeVector(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b,
                          int I, int K, int J) {
  mattTorch::tensor::kernels::cpu::matrixMultTransposeVector(a.getData(), b.getData(), r, I, K, J);
}
void benchBlockTranspose(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b,
                         int I, int K, int J) {
  mattTorch::tensor::kernels::cpu::matrixMultBlockTranspose(a.getData(), b.getData(), r, I, K, J);
}
void benchBlockTransposeVector(double* r, mattTorch::Tensor& a,
                               mattTorch::Tensor& b, int I, int K, int J) {
  mattTorch::tensor::kernels::cpu::matrixMultBlockTransposeVector(a.getData(), b.getData(), r, I, K, J);
}
void benchBlockPackedVector(double* r, mattTorch::Tensor& a,
                            mattTorch::Tensor& b, int I, int K, int J) {
  mattTorch::tensor::kernels::cpu::matrixMultBlockPackedVector(a.getData(), b.getData(), r, I, K, J);
}
void benchOpenBLAS(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int I,
                   int K, int J) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, I, J, K, 1.0,
              a.getData(), K, b.getData(), J, 0.0, r, J);
}
void benchTransposeMulLHS(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b,
                          int I, int K, int J) {
  mattTorch::tensor::kernels::cpu::transposeMultBlockVectorLHS(a.getData(), b.getData(), r, I, K, J);
}
void benchTransposeMulRHS(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b,
                          int I, int K, int J) {
  mattTorch::tensor::kernels::cpu::transposeMultBlockVectorRHS(a.getData(), b.getData(), r, I, K, J);
}

int main() {
  const int I = 1024, K = 1024, J = 1024;
  mattTorch::Tensor a = randomTensor({I, K}, false);
  mattTorch::Tensor b = randomTensor({K, J}, false);

  // Naive / Worse versions have fewer iterations to keep run time sane.
  const int naiveIters = 10;
  runBuffer("MatMul (naive)",
            [&](double* r) { benchNaive(r, a, b, I, K, J); }, I * J,
            naiveIters);
  runBuffer("MatMul (transpose)",
            [&](double* r) { benchTranspose(r, a, b, I, K, J); }, I * J,
            naiveIters);
  runBuffer("MatMul (transpose+SIMD)",
            [&](double* r) { benchTransposeVector(r, a, b, I, K, J); }, I * J,
            naiveIters);
  runBuffer("MatMul (blocked)",
            [&](double* r) { benchBlockTranspose(r, a, b, I, K, J); }, I * J,
            naiveIters);

  // Optimised SIMD + OpenMP variants.
  const int iters = 50;
  runBuffer("MatMul (blocked+SIMD+OpenMP)",
            [&](double* r) { benchBlockTransposeVector(r, a, b, I, K, J); },
            I * J, iters);
  runBuffer("MatMul (MegaKernel)",
            [&](double* r) { benchBlockPackedVector(r, a, b, I, K, J); }, I * J,
            iters);

  // OpenBLAS baseline for comparisson
  runBuffer("MatMul (OpenBLAS)",
            [&](double* r) { benchOpenBLAS(r, a, b, I, K, J); }, I * J, iters);

  // Transpose variants.
  runBuffer("TransposeMul LHS (A^T @ B)",
            [&](double* r) { benchTransposeMulLHS(r, a, b, I, K, J); }, I * J,
            iters);
  runBuffer("TransposeMul RHS (A @ B^T)",
            [&](double* r) { benchTransposeMulRHS(r, a, b, I, K, J); }, I * J,
            iters);
}
