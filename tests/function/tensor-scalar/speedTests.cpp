#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchScalarMul(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a * b;
  result.backward();
}

void benchScalarDiv(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a / b;
  result.backward();
}

void benchScalarAdd(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a + b;
  result.backward();
}

void benchScalarSub(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a - b;
  result.backward();
}

void benchExponent(mattTorch::Tensor& a, int b) {
  mattTorch::Tensor result = a.elementwiseExponent(b);
  result.backward();
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024});
  const double b = randomScalar();

  run("Scalar Mul", [&] { benchScalarMul(a, b); });
  run("Scalar Div", [&] { benchScalarDiv(a, b); });
  run("Scalar Add", [&] { benchScalarAdd(a, b); });
  run("Scalar Sub", [&] { benchScalarSub(a, b); });
  run("Exponent", [&] { benchExponent(a, 3); });
}
