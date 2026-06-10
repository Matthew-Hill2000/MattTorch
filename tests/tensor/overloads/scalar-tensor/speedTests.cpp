#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchScalarAdd(double b, mattTorch::Tensor& a) {
  mattTorch::Tensor r = b + a;
}
void benchScalarSub(double b, mattTorch::Tensor& a) {
  mattTorch::Tensor r = b - a;
}
void benchScalarMul(double b, mattTorch::Tensor& a) {
  mattTorch::Tensor r = b * a;
}
void benchScalarDiv(double b, mattTorch::Tensor& a) {
  mattTorch::Tensor r = b / a;
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024}, false);
  const double b = randomScalar();

  run("Scalar Add", [&] { benchScalarAdd(b, a); });
  run("Scalar Sub", [&] { benchScalarSub(b, a); });
  run("Scalar Mul", [&] { benchScalarMul(b, a); });
  run("Scalar Div", [&] { benchScalarDiv(b, a); });
}
