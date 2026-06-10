#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor-scalar/tensorScalarElementwise.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

bool testTensorScalarAdd(mattTorch::Tensor& a, double b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::tensorScalarAdd(a.getData(), b, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) + b)) > TOL) return false;
  return true;
}

bool testTensorScalarSub(mattTorch::Tensor& a, double b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::tensorScalarSubtract(a.getData(), b, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) - b)) > TOL) return false;
  return true;
}

bool testTensorScalarMul(mattTorch::Tensor& a, double b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::tensorScalarMultiplication(a.getData(), b, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) * b)) > TOL) return false;
  return true;
}

bool testTensorScalarDiv(mattTorch::Tensor& a, double b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::tensorScalarDivision(a.getData(), b, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) / b)) > TOL) return false;
  return true;
}

bool testScalarTensorSub(mattTorch::Tensor& a, double b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::scalarTensorSubtract(a.getData(), b, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (b - a.getValueDirect(i))) > TOL) return false;
  return true;
}

bool testScalarTensorDiv(mattTorch::Tensor& a, double b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::scalarTensorDivision(a.getData(), b, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (b / a.getValueDirect(i))) > TOL) return false;
  return true;
}

bool testExponent(mattTorch::Tensor& a, int e) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::elementwiseExponent(a.getData(), e, r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - std::pow(a.getValueDirect(i), e)) > 1e-6) return false;
  return true;
}

int main() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  const double b = randomScalar();

  run("Tensor-Scalar Add", testTensorScalarAdd(a, b));
  run("Tensor-Scalar Sub", testTensorScalarSub(a, b));
  run("Tensor-Scalar Mul", testTensorScalarMul(a, b));
  run("Tensor-Scalar Div", testTensorScalarDiv(a, b));
  run("Scalar-Tensor Sub", testScalarTensorSub(a, b));
  run("Scalar-Tensor Div", testScalarTensorDiv(a, b));
  run("Exponent", testExponent(a, 3));
}
