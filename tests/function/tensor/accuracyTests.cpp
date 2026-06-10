#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

// ============ Tests ============

bool testReLU(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.ReLU();
  result.backward();

  double* aData = a.getData();
  double* rData = result.getData();
  double* gData = a.getGradientData();

  for (int i = 0; i < a.getNValues(); i++) {
    double expectedFwd = aData[i] > 0 ? aData[i] : 0;
    double expectedGrad = aData[i] > 0 ? 1.0 : 0.0;
    if (std::abs(rData[i] - expectedFwd) > TOL) return false;
    if (std::abs(gData[i] - expectedGrad) > TOL) return false;
  }
  return true;
}

bool testTanh(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.tanh();
  result.backward();

  double* aData = a.getData();
  double* rData = result.getData();
  double* gData = a.getGradientData();

  for (int i = 0; i < a.getNValues(); i++) {
    double expectedFwd = std::tanh(aData[i]);
    double expectedGrad = 1.0 - rData[i] * rData[i];
    if (std::abs(rData[i] - expectedFwd) > TOL) return false;
    if (std::abs(gData[i] - expectedGrad) > TOL) return false;
  }
  return true;
}

bool testMean(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.mean();
  if (result.getNValues() != 1) return false;

  result.backward();

  double* aData = a.getData();
  double* gData = a.getGradientData();
  int n = a.getNValues();

  double sum = 0;
  for (int i = 0; i < n; i++) sum += aData[i];
  if (std::abs(result.getData()[0] - sum / n) > TOL) return false;

  for (int i = 0; i < n; i++) {
    if (std::abs(gData[i] - 1.0 / n) > TOL) return false;
  }
  return true;
}

bool testExp(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.exponential();
  result.backward();

  double* aData = a.getData();
  double* rData = result.getData();
  double* gData = a.getGradientData();

  for (int i = 0; i < a.getNValues(); i++) {
    double expectedFwd = std::exp(aData[i]);
    double expectedGrad = expectedFwd;  // d(e^x)/dx = e^x
    if (std::abs(rData[i] - expectedFwd) > TOL) return false;
    if (std::abs(gData[i] - expectedGrad) > TOL) return false;
  }
  return true;
}

bool testReductionSum(mattTorch::Tensor& a, int dim) {
  mattTorch::Tensor result = a.reductionSum(dim);
  result.backward();

  double* gData = a.getGradientData();

  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(gData[i] - 1.0) > TOL) return false;
  }
  return true;
}

bool testBroadcast(mattTorch::Tensor& a, int pos, int dim) {
  mattTorch::Tensor result = a.broadcast(pos, dim);
  result.backward();

  double* gData = a.getGradientData();

  for (int i = 0; i < a.getNValues(); i++) {
    if (std::abs(gData[i] - double(dim)) > TOL) return false;
  }
  return true;
}

int main() {
  mattTorch::Tensor a1 = randomTensor({1024, 1024});
  mattTorch::Tensor a2 = randomTensor({1024, 1024});
  mattTorch::Tensor a3 = randomTensor({1024, 1024});
  mattTorch::Tensor a4 = randomTensor({1024, 1024});
  mattTorch::Tensor a5 = randomTensor({1024, 1024});
  mattTorch::Tensor small = randomTensor({5, 5});

  run("ReLU", testReLU(a1));
  run("Tanh", testTanh(a2));
  run("Mean", testMean(a3));
  run("Exp", testExp(small));
  run("ReductionSum (dim=0)", testReductionSum(a4, 0));
  run("Broadcast (pos=0, dim=3)", testBroadcast(a5, 0, 3));
}
