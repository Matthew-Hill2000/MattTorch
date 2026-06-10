#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

// ============ Tests ============

bool testScalarAdd(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a + b;
  mattTorch::Tensor grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  return a.detachGradient() == grad;
}

bool testScalarSub(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a - b;
  mattTorch::Tensor grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  return a.detachGradient() == grad;
}

bool testScalarMul(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a * b;
  mattTorch::Tensor grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  mattTorch::Tensor expected(a.getDimensions());
  expected = b;
  return a.detachGradient() == expected;
}

bool testScalarDiv(mattTorch::Tensor& a, double b) {
  mattTorch::Tensor result = a / b;
  mattTorch::Tensor grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  mattTorch::Tensor expected(a.getDimensions());
  expected = 1.0 / b;
  return a.detachGradient() == expected;
}

bool testExponent(mattTorch::Tensor& a, int b) {
  mattTorch::Tensor result = a.elementwiseExponent(b);
  mattTorch::Tensor grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  mattTorch::Tensor expected(a.getDimensions());
  expected.setRequiresGrad(false);

  double* expData = expected.getData();
  double* aData = a.getData();

  for (int i = 0; i < a.getNValues(); i++) {
    expData[i] = b * std::pow(aData[i], b - 1);
  }
  return a.detachGradient() == expected;
}

// ============ Inplace Tests ============

bool testInplaceScalarAdd(mattTorch::Tensor& a, double b) {
  a += b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad;
}

bool testInplaceScalarSub(mattTorch::Tensor& a, double b) {
  a -= b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad;
}

bool testInplaceScalarMul(mattTorch::Tensor& a, double b) {
  a *= b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  mattTorch::Tensor expected(a.getDimensions());
  expected = b;
  return a.detachGradient() == expected;
}

bool testInplaceScalarDiv(mattTorch::Tensor& a, double b) {
  a /= b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  mattTorch::Tensor expected(a.getDimensions());
  expected = 1.0 / b;
  return a.detachGradient() == expected;
}

int main() {
  mattTorch::Tensor a1 = randomTensor({6, 6});
  mattTorch::Tensor a2 = randomTensor({6, 6});
  mattTorch::Tensor a3 = randomTensor({6, 6});
  mattTorch::Tensor a4 = randomTensor({6, 6});
  mattTorch::Tensor a5 = randomTensor({6, 6});
  double b = randomScalar();

  run("Scalar Add", testScalarAdd(a1, b));
  run("Scalar Sub", testScalarSub(a2, b));
  run("Scalar Mul", testScalarMul(a3, b));
  run("Scalar Div", testScalarDiv(a4, b));
  run("Exponent", testExponent(a5, 3));

  mattTorch::Tensor a6 = randomTensor({6, 6});
  mattTorch::Tensor a7 = randomTensor({6, 6});
  mattTorch::Tensor a8 = randomTensor({6, 6});
  mattTorch::Tensor a9 = randomTensor({6, 6});

  run("Inplace Scalar Add", testInplaceScalarAdd(a6, b));
  run("Inplace Scalar Sub", testInplaceScalarSub(a7, b));
  run("Inplace Scalar Mul", testInplaceScalarMul(a8, b));
  run("Inplace Scalar Div", testInplaceScalarDiv(a9, b));
}
