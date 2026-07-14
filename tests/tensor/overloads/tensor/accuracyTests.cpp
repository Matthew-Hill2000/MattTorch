#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

// ============ Tests ============

bool testReLU(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.ReLU();
  for (int i = 0; i < a.getNValues(); i++) {
    double x = a.getValueDirect(i);
    double expected = x > 0 ? x : 0;
    if (result.getValueDirect(i) != expected) return false;
  }
  return true;
}

bool testTanh(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.tanh();
  for (int i = 0; i < a.getNValues(); i++) {
    double expected = std::tanh(a.getValueDirect(i));
    if (std::abs(result.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testMean(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.mean();
  if (result.getNValues() != 1) return false;
  double sum = 0;
  for (int i = 0; i < a.getNValues(); i++) {
    sum += a.getValueDirect(i);
  }
  double expected = sum / a.getNValues();
  return std::abs(result.getValueDirect(0) - expected) <= 1e-6;
}

bool testLog(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.log();
  for (int i = 0; i < a.getNValues(); i++) {
    double expected = std::log(a.getValueDirect(i));
    if (std::abs(result.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testExp(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.exponential();
  for (int i = 0; i < a.getNValues(); i++) {
    double expected = std::exp(a.getValueDirect(i));
    if (std::abs(result.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testReductionSum(mattTorch::Tensor& a, int dim) {
  mattTorch::Tensor result = a.reductionSum(dim);
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

bool testReductionSumAllDims(mattTorch::Tensor& a) {
  for (size_t dim = 0; dim < a.getDimensions().size(); dim++) {
    if (!testReductionSum(a, dim)) return false;
  }
  return true;
}

bool testBroadcast(mattTorch::Tensor& a, int pos, int dim) {
  mattTorch::Tensor result = a.broadcast(pos, dim);
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

bool testBroadcastAllPositions(mattTorch::Tensor& a, int dim) {
  for (size_t pos = 0; pos <= a.getDimensions().size(); pos++) {
    if (!testBroadcast(a, pos, dim)) return false;
  }
  return true;
}

bool testArgmax(mattTorch::Tensor& a) {
  // Independently find the flat index of the maximum element.
  int maxFlat = 0;
  double maxValue = a.getValueDirect(0);
  for (int i = 1; i < a.getNValues(); i++) {
    if (a.getValueDirect(i) > maxValue) {
      maxValue = a.getValueDirect(i);
      maxFlat = i;
    }
  }

  // Unravel the flat index into its row-major multi-dimensional indices.
  auto dims = a.getDimensions();
  mattTorch::Dims expected;
  for (size_t k = 0; k < dims.size(); k++) expected.push_back(0);
  int rem = maxFlat;
  for (int k = static_cast<int>(dims.size()) - 1; k >= 0; k--) {
    expected[k] = rem % dims[k];
    rem /= dims[k];
  }

  return a.argmax() == expected;
}

int main() {
  mattTorch::Tensor a2d = randomTensor({128, 128}, false);
  mattTorch::Tensor a3d = randomTensor({32, 32, 32}, false);
  mattTorch::Tensor small = randomTensor({6, 6}, false);

  run("ReLU", testReLU(a2d));
  run("Tanh", testTanh(a2d));
  run("Log", testLog(a2d));
  run("Exp", testExp(a2d));
  run("Mean", testMean(a2d));
  run("Argmax (2D)", testArgmax(a2d));
  run("Argmax (3D)", testArgmax(a3d));
  run("ReductionSum (all dims)", testReductionSumAllDims(a3d));
  run("Broadcast (all positions)", testBroadcastAllPositions(small, 4));
}
