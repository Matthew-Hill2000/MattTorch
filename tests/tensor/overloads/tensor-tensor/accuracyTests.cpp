#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-4;
constexpr double TOL_MATMUL = 1e-8;

// ============ Tests ============

bool testAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor result = a + b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) + b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor result = a - b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) - b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor result = a * b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) * b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor result = a / b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(result.getValueDirect(i) -
                 (a.getValueDirect(i) / b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testEquality(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aCopy = a.deepCopy();
  if (!(a == aCopy)) return false;

  if (a == b) return false;

  if (a != aCopy) return false;
  if (!(a != b)) return false;

  return true;
}

bool testInplaceAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a += b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) + b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testInplaceSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a -= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) - b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testInplaceMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a *= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) * b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testInplaceDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a /= b;
  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(a.getValueDirect(i) -
                 (aOld.getValueDirect(i) / b.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testMatMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor result = a.matrixMultiply(b);

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

bool testTransposeMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor resultLHS = a.transposeMultiply(b, true);   // A^T * B
  mattTorch::Tensor resultRHS = a.transposeMultiply(b, false);  // A * B^T

  int m = a.getDimensions()[0], k = a.getDimensions()[1],
      n = b.getDimensions()[1];

  mattTorch::Tensor LHS({k, n});
  for (int i = 0; i < k; i++) {
    for (int j = 0; j < n; j++) {
      for (int l = 0; l < m; l++) {
        LHS[{i, j}] += a[{l, i}] * b[{l, j}];
      }
    }
  }

  mattTorch::Tensor RHS({m, n});
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      for (int l = 0; l < k; l++) {
        RHS[{i, j}] += a[{i, l}] * b[{j, l}];
      }
    }
  }

  return LHS == resultLHS && RHS == resultRHS;
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024}, false);
  mattTorch::Tensor b = randomTensor({1024, 1024}, false);

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
