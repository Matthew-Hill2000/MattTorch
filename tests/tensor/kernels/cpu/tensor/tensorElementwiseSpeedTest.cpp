#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/kernels/cpu/tensor/tensorElementwise.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchReLU(double* r, mattTorch::Tensor& in, double* mask, int n) {
  mattTorch::tensor::kernels::cpu::ReLU(in.getData(), r, mask, n);
}
void benchTanh(double* r, mattTorch::Tensor& in, int n) {
  mattTorch::tensor::kernels::cpu::tanh(in.getData(), r, n);
}
void benchLog(double* r, mattTorch::Tensor& in, int n) {
  mattTorch::tensor::kernels::cpu::log(in.getData(), r, n);
}
void benchExp(double* r, mattTorch::Tensor& in, int n) {
  mattTorch::tensor::kernels::cpu::exponential(in.getData(), r, n);
}
void benchMean(double* r, mattTorch::Tensor& in, int n) {
  mattTorch::tensor::kernels::cpu::mean(in.getData(), r, n);
}
void benchBroadcast(double* r, mattTorch::Tensor& in) {
  mattTorch::tensor::kernels::cpu::broadcast(in.getData(), r, in.getNValues(), 1, 10);
}

int main() {
  mattTorch::Tensor large = randomTensor({1024, 1024}, false);
  mattTorch::Tensor small = randomTensor({1024}, false);
  const int n = large.getNValues();

  // ReLU also writes a backward mask
  double* reluMask = createResultBuffer(n);

  runBuffer("ReLU", [&](double* r) { benchReLU(r, large, reluMask, n); }, n);
  runBuffer("Tanh", [&](double* r) { benchTanh(r, large, n); }, n);
  runBuffer("Log", [&](double* r) { benchLog(r, large, n); }, n);
  runBuffer("Exp", [&](double* r) { benchExp(r, large, n); }, n);
  runBuffer("Mean", [&](double* r) { benchMean(r, large, n); }, 1);
  runBuffer(
      "Broadcast", [&](double* r) { benchBroadcast(r, small); },
      small.getNValues() * 10);

  free(reluMask);
}
