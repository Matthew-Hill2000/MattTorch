#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor/tensorElementwise.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

bool testReLU() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  const int n = a.getNValues();
  double* result = createResultBuffer(n);
  double* mask = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::ReLU(a.getData(), result, mask, n);
  for (int i = 0; i < n; i++) {
    const double x = a.getValueDirect(i);
    if (result[i] != (x > 0 ? x : 0.0)) return false;
    if (mask[i] != (x > 0 ? 1.0 : 0.0)) return false;
  }
  return true;
}

bool testTanh() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  const int n = a.getNValues();
  double* result = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::tanh(a.getData(), result, n);
  for (int i = 0; i < n; i++) {
    if (std::abs(result[i] - std::tanh(a.getValueDirect(i))) > TOL)
      return false;
  }
  return true;
}

bool testExp() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  const int n = a.getNValues();
  double* result = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::exponential(a.getData(), result, n);
  for (int i = 0; i < n; i++) {
    if (std::abs(result[i] - std::exp(a.getValueDirect(i))) > TOL) return false;
  }
  return true;
}

bool testLog() {
  // Positive inputs so log is defined.
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  const int n = a.getNValues();
  for (int i = 0; i < n; i++) a.setValueDirect(i, std::abs(a.getValueDirect(i)) + 0.1);
  double* result = createResultBuffer(n);
  mattTorch::tensor::kernels::cpu::log(a.getData(), result, n);
  for (int i = 0; i < n; i++) {
    if (std::abs(result[i] - std::log(a.getValueDirect(i))) > TOL) return false;
  }
  return true;
}

bool testMean() {
  mattTorch::Tensor a = randomTensor({128, 128}, false);
  const int n = a.getNValues();
  double* result = createResultBuffer(1);
  mattTorch::tensor::kernels::cpu::mean(a.getData(), result, n);
  double sum = 0;
  for (int i = 0; i < n; i++) sum += a.getValueDirect(i);
  return std::abs(result[0] - sum / n) <= 1e-6;
}

bool testBroadcast() {
  // Two blocks of size 3, each repeated 4 times: [b0 x4][b1 x4].
  const int blockSize = 3, numBlocks = 2, repeat = 4;
  mattTorch::Tensor a = randomTensor({numBlocks * blockSize}, false);
  double* result = createResultBuffer(numBlocks * repeat * blockSize);
  mattTorch::tensor::kernels::cpu::broadcast(a.getData(), result, blockSize, numBlocks, repeat);
  for (int b = 0; b < numBlocks; b++) {
    for (int rep = 0; rep < repeat; rep++) {
      for (int i = 0; i < blockSize; i++) {
        const double in = a.getValueDirect(b * blockSize + i);
        const double out = result[(b * repeat + rep) * blockSize + i];
        if (std::abs(in - out) > TOL) return false;
      }
    }
  }
  return true;
}

int main() {
  run("ReLU", testReLU());
  run("Tanh", testTanh());
  run("Exp", testExp());
  run("Log", testLog());
  run("Mean", testMean());
  run("Broadcast", testBroadcast());
}
