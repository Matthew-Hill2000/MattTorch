#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchReLU(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.ReLU();
  result.backward();
}

void benchTanh(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.tanh();
  result.backward();
}

void benchReductionSum(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.reductionSum(0);
  result.backward();
}

void benchBroadcast(mattTorch::Tensor& a, int dim) {
  mattTorch::Tensor result = a.broadcast(0, dim);
  result.backward();
}

void benchMean(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.mean();
  result.backward();
}

void benchExp(mattTorch::Tensor& a) {
  mattTorch::Tensor result = a.exponential();
  result.backward();
}

int main() {
  mattTorch::Tensor large = randomTensor({1024, 1024});
  mattTorch::Tensor small = randomTensor({64});

  run("ReLU", [&] { benchReLU(large); });
  run("Exp", [&] { benchExp(large); });
  run("Tanh", [&] { benchTanh(large); });
  run("ReductionSum", [&] { benchReductionSum(large); });
  run("Mean", [&] { benchMean(large); });
  run("Broadcast", [&] { benchBroadcast(large, 1024); });
}
