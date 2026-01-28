#include <mattTorch/mattTorch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

constexpr double TOL = 1e-9;

// Creates a random tensor of given size
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

bool testReLU(mattTorch::TensorView& a) {
  auto result = a.ReLU();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  double* aData = a.getData();
  double* rData = result.getData();
  double* gData = a.detachGradient().getData();

  for (int i = 0; i < a.getNValues(); i++) {
    double expectedFwd = aData[i] > 0 ? aData[i] : 0;
    double expectedGrad = aData[i] > 0 ? 1.0 : 0.0;
    if (std::abs(rData[i] - expectedFwd) > TOL) return false;
    if (std::abs(gData[i] - expectedGrad) > TOL) return false;
  }
  return true;
}

bool testTanh(mattTorch::TensorView& a) {
  auto result = a.tanh();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  double* aData = a.getData();
  double* rData = result.getData();
  double* gData = a.detachGradient().getData();

  for (int i = 0; i < a.getNValues(); i++) {
    double expectedFwd = std::tanh(aData[i]);
    double expectedGrad = 1.0 - rData[i] * rData[i];
    if (std::abs(rData[i] - expectedFwd) > TOL) return false;
    if (std::abs(gData[i] - expectedGrad) > TOL) return false;
  }
  return true;
}

bool testMean(mattTorch::TensorView& a) {
  auto result = a.mean();
  if (result.getNValues() != 1) return false;

  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  double* aData = a.getData();
  double* gData = a.detachGradient().getData();
  int n = a.getNValues();

  double sum = 0;
  for (int i = 0; i < n; i++) sum += aData[i];
  if (std::abs(result.getData()[0] - sum / n) > 1e-6) return false;

  for (int i = 0; i < n; i++) {
    if (std::abs(gData[i] - 1.0 / n) > TOL) return false;
  }
  return true;
}

bool testExp(mattTorch::TensorView& a) {
  auto result = a.exponential();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  double* aData = a.getData();
  double* rData = result.getData();
  double* gData = a.detachGradient().getData();

  for (int i = 0; i < a.getNValues(); i++) {
    double expectedFwd = std::exp(aData[i]);
    double expectedGrad = expectedFwd;  // d(e^x)/dx = e^x
    if (std::abs(rData[i] - expectedFwd) > TOL) return false;
    if (std::abs(gData[i] - expectedGrad) > TOL) return false;
  }
  return true;
}

bool testReductionSum(mattTorch::TensorView& a, int dim) {
  auto result = a.reductionSum(dim);
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  double* gData = a.detachGradient().getData();

  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(gData[i] - 1.0) > TOL) return false;
  }
  return true;
}

bool testBroadcast(mattTorch::TensorView& a, int pos, int dim) {
  auto result = a.broadcast(pos, dim);
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);

  double* gData = a.detachGradient().getData();

  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(gData[i] - double(dim)) > TOL) return false;
  }
  return true;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a1 = randomTensor(1024, 1024);
  auto a2 = randomTensor(1024, 1024);
  auto a3 = randomTensor(1024, 1024);
  auto a4 = randomTensor(1024, 1024);
  auto a5 = randomTensor(1024, 1024);
  auto small = randomTensor(5, 5);

  run("ReLU", testReLU(a1));
  run("Tanh", testTanh(a2));
  run("Mean", testMean(a3));
  run("Exp", testExp(small));
  run("ReductionSum (dim=0)", testReductionSum(a4, 0));
  run("Broadcast (pos=0, dim=3)", testBroadcast(a5, 0, 3));
}
