#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor-scalar/tensorScalarElementwise.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchTensorScalarAdd(double* r, mattTorch::Tensor& a, double b, int n) {
  mattTorch::tensor::kernels::cpu::tensorScalarAdd(a.getData(), b, r, n);
}
void benchTensorScalarSub(double* r, mattTorch::Tensor& a, double b, int n) {
  mattTorch::tensor::kernels::cpu::tensorScalarSubtract(a.getData(), b, r, n);
}
void benchTensorScalarMul(double* r, mattTorch::Tensor& a, double b, int n) {
  mattTorch::tensor::kernels::cpu::tensorScalarMultiplication(a.getData(), b, r, n);
}
void benchTensorScalarDiv(double* r, mattTorch::Tensor& a, double b, int n) {
  mattTorch::tensor::kernels::cpu::tensorScalarDivision(a.getData(), b, r, n);
}
void benchScalarTensorSub(double* r, mattTorch::Tensor& a, double b, int n) {
  mattTorch::tensor::kernels::cpu::scalarTensorSubtract(a.getData(), b, r, n);
}
void benchScalarTensorDiv(double* r, mattTorch::Tensor& a, double b, int n) {
  mattTorch::tensor::kernels::cpu::scalarTensorDivision(a.getData(), b, r, n);
}
void benchExponent(double* r, mattTorch::Tensor& a, int n) {
  mattTorch::tensor::kernels::cpu::elementwiseExponent(a.getData(), 3, r, n);
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024}, false);
  const double b = randomScalar();
  const int n = a.getNValues();

  runBuffer("Tensor-Scalar Add",
            [&](double* r) { benchTensorScalarAdd(r, a, b, n); }, n);
  runBuffer("Tensor-Scalar Sub",
            [&](double* r) { benchTensorScalarSub(r, a, b, n); }, n);
  runBuffer("Tensor-Scalar Mul",
            [&](double* r) { benchTensorScalarMul(r, a, b, n); }, n);
  runBuffer("Tensor-Scalar Div",
            [&](double* r) { benchTensorScalarDiv(r, a, b, n); }, n);
  runBuffer("Scalar-Tensor Sub",
            [&](double* r) { benchScalarTensorSub(r, a, b, n); }, n);
  runBuffer("Scalar-Tensor Div",
            [&](double* r) { benchScalarTensorDiv(r, a, b, n); }, n);
  runBuffer("Exponent", [&](double* r) { benchExponent(r, a, n); }, n);
}
