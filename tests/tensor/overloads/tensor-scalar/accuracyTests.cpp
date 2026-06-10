#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-10;

// ============ Tests ============

bool testScalarAdd(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a + b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) + b)) > TOL)
      return false;
  }
  return true;
}

bool testScalarSub(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a - b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) - b)) > TOL)
      return false;
  }
  return true;
}

bool testScalarMul(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a * b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) * b)) > TOL)
      return false;
  }
  return true;
}

bool testScalarDiv(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a / b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) / b)) > TOL)
      return false;
  }
  return true;
}


bool testInplaceScalarAdd(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a += b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) - (aOld.getValueDirect(i) + b)) > TOL)
      return false;
  }
  return true;
}

bool testInplaceScalarSub(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a -= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) - (aOld.getValueDirect(i) - b)) > TOL)
      return false;
  }
  return true;
}

bool testInplaceScalarMul(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a *= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) - (aOld.getValueDirect(i) * b)) > TOL)
      return false;
  }
  return true;
}

bool testInplaceScalarDiv(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a /= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) - (aOld.getValueDirect(i) / b)) > TOL)
      return false;
  }
  return true;
}

bool testExponent(mattTorch::Tensor& a, int b) {
  mattTorch::Tensor result = a.elementwiseExponent(b);
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - std::pow(a.getValueDirect(i), b)) >
        TOL)
      return false;
  }
  return true;
}

bool testScalarAssign(mattTorch::Tensor& a, int b) {
  a = b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) - b) > TOL) return false;
  }
  return true;
}
int main() {
  mattTorch::Tensor a = randomTensor({8, 8}, false);
  double b = randomScalar();

  run("Scalar Add", testScalarAdd(a, b));
  run("Scalar Sub", testScalarSub(a, b));
  run("Scalar Mul", testScalarMul(a, b));
  run("Scalar Div", testScalarDiv(a, b));
  run("InplaceScalarMul", testInplaceScalarMul(a, b));
  run("InplaceScalarAdd", testInplaceScalarAdd(a, b));
  run("InplaceScalarSub", testInplaceScalarSub(a, b));
  run("InplaceScalarDiv", testInplaceScalarDiv(a, b));
  run("Exponent", testExponent(a, 3));
  run("Scalar Assign", testScalarAssign(a, b));
}
