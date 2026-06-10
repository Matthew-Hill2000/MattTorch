#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/elementwise.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

bool testAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::elementwiseAdd(a.getData(), b.getData(), r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) + b.getValueDirect(i))) > TOL)
      return false;
  return true;
}

bool testSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::elementwiseSubtract(a.getData(), b.getData(), r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) - b.getValueDirect(i))) > TOL)
      return false;
  return true;
}

bool testMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::elementwiseMultiplication(a.getData(), b.getData(), r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) * b.getValueDirect(i))) > TOL)
      return false;
  return true;
}

bool testDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  const int n = a.getNValues();
  double* r = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::elementwiseDivision(a.getData(), b.getData(), r, n);
  for (int i = 0; i < n; i++)
    if (std::abs(r[i] - (a.getValueDirect(i) / b.getValueDirect(i))) > TOL)
      return false;
  return true;
}

bool testReductionSum() {
  // Reduce the middle dim of a {4,5,3} tensor: outer=4, reduce=5, inner=3.
  mattTorch::Tensor a = randomTensor({4, 5, 3}, false);
  const int outer = 4, reduce = 5, inner = 3;
  double* r = createResultBuffer(outer * inner);
  mattTorch::tensor::kernels::cpu::reductionSum(a.getData(), r, outer, reduce, inner);
  for (int o = 0; o < outer; o++) {
    for (int in = 0; in < inner; in++) {
      double sum = 0;
      for (int red = 0; red < reduce; red++)
        sum += a.getValueDirect(o * reduce * inner + red * inner + in);
      if (std::abs(r[o * inner + in] - sum) > 1e-6) return false;
    }
  }
  return true;
}

bool testInplaceAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aCopy = a.deepCopy();
  const int n = a.getNValues();
  mattTorch::tensor::kernels::cpu::inplaceElementwiseAdd(aCopy.getData(), b.getData(), n);
  for (int i = 0; i < n; i++)
    if (std::abs(aCopy.getValueDirect(i) -
                 (a.getValueDirect(i) + b.getValueDirect(i))) > TOL)
      return false;
  return true;
}

int main() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  mattTorch::Tensor b = randomTensor({128, 128}, false);

  run("Add", testAdd(a, b));
  run("Sub", testSub(a, b));
  run("Mul", testMul(a, b));
  run("Div", testDiv(a, b));
  run("ReductionSum", testReductionSum());
  run("Inplace Add", testInplaceAdd(a, b));
}
