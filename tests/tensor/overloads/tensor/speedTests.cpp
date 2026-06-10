#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchReLU(mattTorch::Tensor& a) { mattTorch::Tensor r = a.ReLU(); }
void benchTanh(mattTorch::Tensor& a) { mattTorch::Tensor r = a.tanh(); }
void benchLog(mattTorch::Tensor& a) { mattTorch::Tensor r = a.log(); }
void benchExp(mattTorch::Tensor& a) { mattTorch::Tensor r = a.exponential(); }
void benchReductionSum(mattTorch::Tensor& a) {
  mattTorch::Tensor r = a.reductionSum(1);
}
void benchMean(mattTorch::Tensor& a) { mattTorch::Tensor r = a.mean(); }
void benchBroadcast(mattTorch::Tensor& a) {
  mattTorch::Tensor r = a.broadcast(0, 10);
}

int main() {
  mattTorch::Tensor large = randomTensor({1024, 1024}, false);
  mattTorch::Tensor small = randomTensor({1024}, false);

  run("ReLU", [&] { benchReLU(large); });
  run("Tanh", [&] { benchTanh(large); });
  run("Log", [&] { benchLog(large); });
  run("Exp", [&] { benchExp(large); });
  run("ReductionSum", [&] { benchReductionSum(large); });
  run("Mean", [&] { benchMean(large); });
  run("Broadcast", [&] { benchBroadcast(small); });
}
