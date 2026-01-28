#include <mattTorch/mattTorch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

constexpr double TOL = 1e-9;

mattTorch::TensorView randomTensor(std::vector<int> dims) {
  static unsigned seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  mattTorch::TensorView t(dims);
  t.setRequiresGrad(false);
  double* data = t.getData();
  for (int i = 0; i < t.getNValues(); i++) {
    data[i] = dist(gen);
  }
  return t;
}

// ============ Tests ============

bool testReLU(mattTorch::TensorView& a) {
  auto result = a.ReLU();
  for (int i = 0; i < a.getNValues(); i++) {
    double x = a.getValueDirect(i);
    double expected = x > 0 ? x : 0;
    if (result.getValueDirect(i) != expected) return false;
  }
  return true;
}

bool testTanh(mattTorch::TensorView& a) {
  auto result = a.tanh();
  for (int i = 0; i < a.getNValues(); i++) {
    double expected = std::tanh(a.getValueDirect(i));
    if (std::abs(result.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testMean(mattTorch::TensorView& a) {
  auto result = a.mean();
  if (result.getNValues() != 1) return false;
  double sum = 0;
  for (int i = 0; i < a.getNValues(); i++) {
    sum += a.getValueDirect(i);
  }
  double expected = sum / a.getNValues();
  return std::abs(result.getValueDirect(0) - expected) <= 1e-6;
}

bool testLog(mattTorch::TensorView& a) {
  auto result = a.log();
  for (int i = 0; i < a.getNValues(); i++) {
    double expected = std::log(a.getValueDirect(i));
    if (std::abs(result.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testExp(mattTorch::TensorView& a) {
  auto result = a.exponential();
  for (int i = 0; i < a.getNValues(); i++) {
    double expected = std::exp(a.getValueDirect(i));
    if (std::abs(result.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testReductionSum(mattTorch::TensorView& a, int dim) {
  auto result = a.reductionSum(dim);
  auto dims = a.getDimensions();
  int outerSize = 1, innerSize = 1;
  for (int i = 0; i < dim; i++) outerSize *= dims[i];
  for (size_t i = dim + 1; i < dims.size(); i++) innerSize *= dims[i];
  int reduceSize = dims[dim];
  for (int o = 0; o < outerSize; o++) {
    for (int in = 0; in < innerSize; in++) {
      double sum = 0;
      for (int r = 0; r < reduceSize; r++) {
        int idx = o * reduceSize * innerSize + r * innerSize + in;
        sum += a.getValueDirect(idx);
      }
      int resultIdx = o * innerSize + in;
      if (std::abs(result.getValueDirect(resultIdx) - sum) > 1e-4) return false;
    }
  }
  return true;
}

bool testReductionSumAllDims(mattTorch::TensorView& a) {
  for (size_t dim = 0; dim < a.getDimensions().size(); dim++) {
    if (!testReductionSum(a, dim)) return false;
  }
  return true;
}

bool testBroadcast(mattTorch::TensorView& a, int pos, int dim) {
  auto result = a.broadcast(pos, dim);
  auto aDims = a.getDimensions();
  int outerSize = 1, innerSize = 1;
  for (int i = 0; i < pos; i++) outerSize *= aDims[i];
  for (size_t i = pos; i < aDims.size(); i++) innerSize *= aDims[i];
  for (int o = 0; o < outerSize; o++) {
    for (int rep = 0; rep < dim; rep++) {
      for (int in = 0; in < innerSize; in++) {
        int aIdx = o * innerSize + in;
        int rIdx = (o * dim + rep) * innerSize + in;
        if (std::abs(a.getValueDirect(aIdx) - result.getValueDirect(rIdx)) >
            TOL)
          return false;
      }
    }
  }
  return true;
}

bool testBroadcastAllPositions(mattTorch::TensorView& a, int dim) {
  for (size_t pos = 0; pos <= a.getDimensions().size(); pos++) {
    if (!testBroadcast(a, pos, dim)) return false;
  }
  return true;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a2d = randomTensor({128, 128});
  auto a3d = randomTensor({32, 32, 32});
  auto small = randomTensor({20, 20});

  run("ReLU", testReLU(a2d));
  run("Tanh", testTanh(a2d));
  run("Log", testLog(a2d));
  run("Exp", testExp(a2d));
  run("Mean", testMean(a2d));
  run("ReductionSum (all dims)", testReductionSumAllDims(a3d));
  run("Broadcast (all positions)", testBroadcastAllPositions(small, 8));
}
