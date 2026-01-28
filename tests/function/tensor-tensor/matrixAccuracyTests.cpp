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

bool testMatMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a.matrixMultiply(b);
  mattTorch::TensorView grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::TensorView aGrad(a.getDimensions());
  mattTorch::TensorView bGrad(b.getDimensions());

  int m = a.getDimensions()[0], k = a.getDimensions()[1],
      n = b.getDimensions()[1];

  for (int i = 0; i < m; i++)
    for (int j = 0; j < k; j++)
      for (int l = 0; l < n; l++) aGrad[{i, j}] += grad[{i, l}] * b[{j, l}];

  for (int i = 0; i < k; i++)
    for (int j = 0; j < n; j++)
      for (int l = 0; l < m; l++) bGrad[{i, j}] += a[{l, i}] * grad[{l, j}];

  return a.detachGradient() == aGrad && b.detachGradient() == bGrad;
}

bool testTransposeLHS(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a.transposeMultiply(b, true);
  mattTorch::TensorView grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::TensorView aGrad(a.getDimensions());
  mattTorch::TensorView bGrad(b.getDimensions());

  int m = a.getDimensions()[0], n = a.getDimensions()[1],
      k = b.getDimensions()[1];

  for (int p = 0; p < m; p++)
    for (int q = 0; q < n; q++)
      for (int j = 0; j < k; j++) aGrad[{p, q}] += grad[{q, j}] * b[{p, j}];

  for (int p = 0; p < m; p++)
    for (int q = 0; q < k; q++)
      for (int i = 0; i < n; i++) bGrad[{p, q}] += a[{p, i}] * grad[{i, q}];

  return a.detachGradient() == aGrad && b.detachGradient() == bGrad;
}

bool testTransposeRHS(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a.transposeMultiply(b, false);
  mattTorch::TensorView grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::TensorView aGrad(a.getDimensions());
  mattTorch::TensorView bGrad(b.getDimensions());

  int m = a.getDimensions()[0], n = a.getDimensions()[1],
      k = b.getDimensions()[0];

  for (int p = 0; p < m; p++)
    for (int q = 0; q < n; q++)
      for (int j = 0; j < k; j++) aGrad[{p, q}] += grad[{p, j}] * b[{j, q}];

  for (int p = 0; p < k; p++)
    for (int q = 0; q < n; q++)
      for (int i = 0; i < m; i++) bGrad[{p, q}] += grad[{i, p}] * a[{i, q}];

  return a.detachGradient() == aGrad && b.detachGradient() == bGrad;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a1 = randomTensor(128, 128);
  auto b1 = randomTensor(128, 128);

  auto a2 = randomTensor(128, 64);
  auto b2 = randomTensor(128, 96);

  auto a3 = randomTensor(128, 64);
  auto b3 = randomTensor(96, 64);

  run("MatMul", testMatMul(a1, b1));
  run("Transpose LHS (A^T * B)", testTransposeLHS(a2, b2));
  run("Transpose RHS (A * B^T)", testTransposeRHS(a3, b3));
}
