#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchScalarAdd(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor r = a + b;
}
void benchScalarSub(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor r = a - b;
}
void benchScalarMul(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor r = a * b;
}
void benchScalarDiv(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor r = a / b;
}
void benchExponent(mattTorch::Tensor& a) {
  mattTorch::Tensor r = a.elementwiseExponent(3);
}
void benchScalarAssign(mattTorch::Tensor& a, double b) { a = b; }

void benchInplaceScalarAdd(mattTorch::Tensor& a, double b) { a += b; }
void benchInplaceScalarSub(mattTorch::Tensor& a, double b) { a -= b; }
void benchInplaceScalarMul(mattTorch::Tensor& a, double b) { a *= b; }
void benchInplaceScalarDiv(mattTorch::Tensor& a, double b) { a /= b; }

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024}, false);
  const double b = randomScalar();

  run("Scalar Add", [&] { benchScalarAdd(a, b); });
  run("Scalar Sub", [&] { benchScalarSub(a, b); });
  run("Scalar Mul", [&] { benchScalarMul(a, b); });
  run("Scalar Div", [&] { benchScalarDiv(a, b); });
  run("Exponent", [&] { benchExponent(a); });
  run("Scalar Assign", [&] { benchScalarAssign(a, b); });

  run("Inplace Scalar Add", [&] { benchInplaceScalarAdd(a, b); });
  run("Inplace Scalar Sub", [&] { benchInplaceScalarSub(a, b); });
  run("Inplace Scalar Mul", [&] { benchInplaceScalarMul(a, b); });
  run("Inplace Scalar Div", [&] { benchInplaceScalarDiv(a, b); });
}
