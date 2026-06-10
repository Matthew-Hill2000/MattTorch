#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/elementwise.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchAdd(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int n) {
  mattTorch::tensor::kernels::cpu::elementwiseAdd(a.getData(), b.getData(), r, n);
}
void benchSub(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int n) {
  mattTorch::tensor::kernels::cpu::elementwiseSubtract(a.getData(), b.getData(), r, n);
}
void benchMul(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int n) {
  mattTorch::tensor::kernels::cpu::elementwiseMultiplication(a.getData(), b.getData(), r, n);
}
void benchDiv(double* r, mattTorch::Tensor& a, mattTorch::Tensor& b, int n) {
  mattTorch::tensor::kernels::cpu::elementwiseDivision(a.getData(), b.getData(), r, n);
}
void benchReductionSum(double* r, mattTorch::Tensor& a, int outer, int reduce) {
  mattTorch::tensor::kernels::cpu::reductionSum(a.getData(), r, outer, reduce, 1);
}
void benchInplaceAdd(mattTorch::Tensor& a, mattTorch::Tensor& b, int n) {
  mattTorch::tensor::kernels::cpu::inplaceElementwiseAdd(a.getData(), b.getData(), n);
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024}, false);
  mattTorch::Tensor b = randomTensor({1024, 1024}, false);
  const int n = a.getNValues();

  runBuffer("Add", [&](double* r) { benchAdd(r, a, b, n); }, n);
  runBuffer("Sub", [&](double* r) { benchSub(r, a, b, n); }, n);
  runBuffer("Mul", [&](double* r) { benchMul(r, a, b, n); }, n);
  runBuffer("Div", [&](double* r) { benchDiv(r, a, b, n); }, n);

  const auto dims = a.getDimensions();
  runBuffer(
      "ReductionSum",
      [&](double* r) { benchReductionSum(r, a, dims[0], dims[1]); }, dims[0]);

  run("Inplace Add", [&] { benchInplaceAdd(a, b, n); });
}
