#include <mattTorch/mattTorch.h>

#include <chrono>
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

// ============ Tests ============

bool testAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a + b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad;
}

bool testSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a - b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad * -1.0;
}

bool testMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a * b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);

  return a.detachGradient() == b && b.detachGradient() == a;
}

bool testDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a / b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::TensorView expectedA = 1.0 / b;
  mattTorch::TensorView expectedBDenom = b * b;
  mattTorch::TensorView expectedB = -1.0 * a / expectedBDenom;

  return a.detachGradient() == expectedA && b.detachGradient() == expectedB;
}

// ============ Inplace Tests ============

bool testInplaceAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  a += b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad;
}

bool testInplaceSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  a -= b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad * -1.0;
}

bool testInplaceMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aOld = a.deepCopy();
  a *= b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == b && b.detachGradient() == aOld;
}

bool testInplaceDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aOld = a.deepCopy();
  a /= b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  mattTorch::TensorView expectedA = 1.0 / b;
  mattTorch::TensorView expectedBDenom = b * b;
  mattTorch::TensorView expectedB = -1.0 * aOld / expectedBDenom;

  return a.detachGradient() == expectedA && b.detachGradient() == expectedB;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a1 = randomTensor(128, 128);
  auto b1 = randomTensor(128, 128);
  auto a2 = randomTensor(128, 128);
  auto b2 = randomTensor(128, 128);
  auto a3 = randomTensor(128, 128);
  auto b3 = randomTensor(128, 128);
  auto a4 = randomTensor(128, 128);
  auto b4 = randomTensor(128, 128);

  run("Add", testAdd(a1, b1));
  run("Sub", testSub(a2, b2));
  run("Mul", testMul(a3, b3));
  run("Div", testDiv(a4, b4));

  auto a5 = randomTensor(128, 128);
  auto b5 = randomTensor(128, 128);
  auto a6 = randomTensor(128, 128);
  auto b6 = randomTensor(128, 128);
  auto a7 = randomTensor(128, 128);
  auto b7 = randomTensor(128, 128);
  auto a8 = randomTensor(128, 128);
  auto b8 = randomTensor(128, 128);

  run("Inplace Add", testInplaceAdd(a5, b5));
  run("Inplace Sub", testInplaceSub(a6, b6));
  run("Inplace Mul", testInplaceMul(a7, b7));
  run("Inplace Div", testInplaceDiv(a8, b8));
}
