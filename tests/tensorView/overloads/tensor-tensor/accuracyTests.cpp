#include <mattTorch/mattTorch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

constexpr double TOL = 1e-4;
constexpr double TOL_MATMUL = 1e-8;

mattTorch::TensorView randomTensor(int rows, int cols) {
  static unsigned seed =
      std::chrono::system_clock::now().time_since_epoch().count();
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

// ============ Tests ============

bool testAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a + b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) + b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a - b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) - b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a * b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) * b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a / b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) / b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testEquality(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aCopy = a.deepCopy();
  if (!(a == aCopy)) return false;

  if (a == b) return false;

  if (a != aCopy) return false;
  if (!(a != b)) return false;

  return true;
}

bool testInplaceAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aOld = a.deepCopy();
  a += b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) + b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testInplaceSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aOld = a.deepCopy();
  a -= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) - b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testInplaceMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aOld = a.deepCopy();
  a *= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) * b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testInplaceDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto aOld = a.deepCopy();
  a /= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) / b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testMatMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a.matrixMultiply(b);
  int m = a.getDimensions()[0], k = a.getDimensions()[1],
      n = b.getDimensions()[1];

  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      double acc = 0;
      for (int l = 0; l < k; l++) {
        acc += a[{i, l}] * b[{l, j}];
      }
      if (std::abs(acc - result[{i, j}]) > TOL_MATMUL) return false;
    }
  }
  return true;
}

bool testTransposeMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto resultLHS = a.transposeMultiply(b, true);   // A^T * B
  auto resultRHS = a.transposeMultiply(b, false);  // A * B^T

  int m = a.getDimensions()[0], k = a.getDimensions()[1],
      n = b.getDimensions()[1];

  mattTorch::TensorView LHS({k, n});
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      for (int l = 0; l < m; l++) {
        LHS[{i, j}] += a[{l, i}] * b[{l, j}];
      }
    }
  }

  mattTorch::TensorView RHS({m, n});
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int l = 0; l < k; l++) {
        RHS[{i, j}] += a[{i, l}] * b[{j, l}];
      }
    }
  }

  return LHS == resultLHS && RHS == resultRHS;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a = randomTensor(128, 128);
  auto b = randomTensor(128, 128);

  run("Add", testAdd(a, b));
  run("Sub", testSub(a, b));
  run("Mul", testMul(a, b));
  run("Div", testDiv(a, b));
  run("Equality", testEquality(a, b));
  run("InplaceAdd", testInplaceAdd(a, b));
  run("InplaceSub", testInplaceSub(a, b));
  run("InplaceMul", testInplaceMul(a, b));
  run("InplaceDiv", testInplaceDiv(a, b));
  run("MatMul", testMatMul(a, b));
  run("TransposeMul", testTransposeMul(a, b));
}
