#include <mattTorch/mattTorch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

mattTorch::TensorView randomTensor(int rows, int cols) {
  static unsigned seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);

  mattTorch::TensorView t({rows, cols});
  double* data = t.getData();
  for (int i = 0; i < t.getNValues(); i++) {
    data[i] = dist(gen);
  }
  return t;
}

double randomScalar() {
  static unsigned seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  return dist(gen);
}

// ============ Tests ============

bool testScalarAdd(mattTorch::TensorView& a, double b) {
  auto result = a + b;
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  return a.detachGradient() == grad;
}

bool testScalarSub(mattTorch::TensorView& a, double b) {
  auto result = a - b;
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  return a.detachGradient() == grad;
}

bool testScalarMul(mattTorch::TensorView& a, double b) {
  auto result = a * b;
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  mattTorch::TensorView expected(a.getDimensions());
  expected = b;
  return a.detachGradient() == expected;
}

bool testScalarDiv(mattTorch::TensorView& a, double b) {
  auto result = a / b;
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  mattTorch::TensorView expected(a.getDimensions());
  expected = 1.0 / b;
  return a.detachGradient() == expected;
}

bool testExponent(mattTorch::TensorView& a, int b) {
  auto result = a.elementwiseExponent(b);
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  mattTorch::TensorView expected(a.getDimensions());
  expected.setRequiresGrad(false);
  double* expData = expected.getData();
  double* aData = a.getData();
  for (int i = 0; i < a.getNValues(); i++) {
    expData[i] = b * std::pow(aData[i], b - 1);
  }
  return a.detachGradient() == expected;
}

// ============ Inplace Tests ============

bool testInplaceScalarAdd(mattTorch::TensorView& a, double b) {
  a += b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad;
}

bool testInplaceScalarSub(mattTorch::TensorView& a, double b) {
  a -= b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad;
}

bool testInplaceScalarMul(mattTorch::TensorView& a, double b) {
  a *= b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  mattTorch::TensorView expected(a.getDimensions());
  expected = b;
  return a.detachGradient() == expected;
}

bool testInplaceScalarDiv(mattTorch::TensorView& a, double b) {
  a /= b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  mattTorch::TensorView expected(a.getDimensions());
  expected = 1.0 / b;
  return a.detachGradient() == expected;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a1 = randomTensor(6, 6);
  auto a2 = randomTensor(6, 6);
  auto a3 = randomTensor(6, 6);
  auto a4 = randomTensor(6, 6);
  auto a5 = randomTensor(6, 6);
  double b = randomScalar();

  run("Scalar Add", testScalarAdd(a1, b));
  run("Scalar Sub", testScalarSub(a2, b));
  run("Scalar Mul", testScalarMul(a3, b));
  run("Scalar Div", testScalarDiv(a4, b));
  run("Exponent", testExponent(a5, 3));

  auto a6 = randomTensor(6, 6);
  auto a7 = randomTensor(6, 6);
  auto a8 = randomTensor(6, 6);
  auto a9 = randomTensor(6, 6);

  run("Inplace Scalar Add", testInplaceScalarAdd(a6, b));
  run("Inplace Scalar Sub", testInplaceScalarSub(a7, b));
  run("Inplace Scalar Mul", testInplaceScalarMul(a8, b));
  run("Inplace Scalar Div", testInplaceScalarDiv(a9, b));
}
