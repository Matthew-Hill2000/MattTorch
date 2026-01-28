#include <mattTorch/mattTorch.h>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

constexpr double TOL = 1e-4;

mattTorch::TensorView randomTensor(int rows, int cols) {
  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  
  mattTorch::TensorView t({rows, cols});
  t.setRequiresGrad(false);
  double* data = t.getData();
  for (int i = 0; i < t.getNValues(); i++) {
    data[i] = dist(gen);
  }
  return t;
}

double randomScalar() {
  static unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  return dist(gen);
}

// ============ Tests ============

bool testScalarAdd(mattTorch::TensorView& a, double b) {
  auto result = b + a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) + b)) > TOL) return false;
  }
  return true;
}

bool testScalarSub(mattTorch::TensorView& a, double b) {
  auto result = b - a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (b - a.getValueDirect(i))) > TOL) return false;
  }
  return true;
}

bool testScalarMul(mattTorch::TensorView& a, double b) {
  auto result = b * a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (a.getValueDirect(i) * b)) > TOL) return false;
  }
  return true;
}

bool testScalarDiv(mattTorch::TensorView& a, double b) {
  auto result = b / a;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) - (b / a.getValueDirect(i))) > TOL) return false;
  }
  return true;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a = randomTensor(128, 128);
  double b = randomScalar();
  
  run("Scalar Add", testScalarAdd(a, b));
  run("Scalar Sub", testScalarSub(a, b));
  run("Scalar Mul", testScalarMul(a, b));
  run("Scalar Div", testScalarDiv(a, b));
}
