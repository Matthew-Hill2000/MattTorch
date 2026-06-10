#include <cblas.h>
#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchMatMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.matrixMultiply(b);
}

void benchOpenBLAS(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int M,
                   int K, int N) {
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1.0,
              a.getData(), K, b.getData(), N, 0.0, r, N);
}

void benchTransposeMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.transposeMultiply(b, false);
}

void benchAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a + b;
}
void benchMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a * b;
}
void benchSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a - b;
}
void benchDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a / b;
}
void benchEquality(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  bool eq = (a == b);
  (void)eq;
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024}, false);
  mattTorch::Tensor b = randomTensor({1024, 1024}, false);

  run("MatMul", [&] { benchMatMul(a, b); }, 50);

  const int M = 1024, K = 1024, N = 1024;
  runBuffer(
      "MatMul (OpenBLAS)", [&](double* r) { benchOpenBLAS(r, a, b, M, K, N); },
      M * N, 50);

  run("TransposeMul", [&] { benchTransposeMul(a, b); }, 50);
  run("Add", [&] { benchAdd(a, b); });
  run("Mul", [&] { benchMul(a, b); });
  run("Sub", [&] { benchSub(a, b); });
  run("Div", [&] { benchDiv(a, b); });
  run("Equality", [&] { benchEquality(a, b); });
}
