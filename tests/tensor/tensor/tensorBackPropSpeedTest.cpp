#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchChainedBackward(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a + b;
  mattTorch::Tensor d = c * a;
  mattTorch::Tensor e = d - b;
  mattTorch::Tensor grad(e.getDimensions());
  grad = 1.0;
  e.backward(grad);
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024});
  mattTorch::Tensor b = randomTensor({1024, 1024});

  run("Backward (chained graph)", [&] { benchChainedBackward(a, b); });
}
