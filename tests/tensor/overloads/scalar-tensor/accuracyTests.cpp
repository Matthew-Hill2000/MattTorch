#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-4;

// ============ Tests ============

bool testScalarAdd(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = b + a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) + b)) > TOL)
      return false;
  }
  return true;
}

bool testScalarSub(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = b - a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (b - a.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testScalarMul(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = b * a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) * b)) > TOL)
      return false;
  }
  return true;
}

bool testScalarDiv(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = b / a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (b / a.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

int main() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  double b = randomScalar();

  run("Scalar Add", testScalarAdd(a, b));
  run("Scalar Sub", testScalarSub(a, b));
  run("Scalar Mul", testScalarMul(a, b));
  run("Scalar Div", testScalarDiv(a, b));
}
