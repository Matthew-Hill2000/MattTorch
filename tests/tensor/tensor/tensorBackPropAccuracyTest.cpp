#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

bool testBackwardPopulatesLeafGrads() {
  mattTorch::Tensor a = randomTensor({4, 4});
  mattTorch::Tensor b = randomTensor({4, 4});
  mattTorch::Tensor c = a + b;
  mattTorch::Tensor grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);
  mattTorch::Tensor ga = a.detachGradient();
  mattTorch::Tensor gb = b.detachGradient();
  for (int i = 0; i < a.getNValues(); i++)
    if (ga.getValueDirect(i) != 1.0 || gb.getValueDirect(i) != 1.0)
      return false;
  return true;
}

bool testGradientAccumulates() {
  mattTorch::Tensor a = randomTensor({4, 4});
  mattTorch::Tensor b = randomTensor({4, 4});
  for (int pass = 0; pass < 2; pass++) {
    mattTorch::Tensor c = a + b;
    mattTorch::Tensor grad(c.getDimensions());
    grad = 1.0;
    c.backward(grad);
  }
  mattTorch::Tensor ga = a.detachGradient();
  for (int i = 0; i < a.getNValues(); i++)
    if (ga.getValueDirect(i) != 2.0) return false;
  return true;
}

bool testResetGradient() {
  mattTorch::Tensor a = randomTensor({4, 4});
  mattTorch::Tensor b = randomTensor({4, 4});
  mattTorch::Tensor c = a + b;
  mattTorch::Tensor grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);
  a.resetGradient();
  mattTorch::Tensor ga = a.detachGradient();
  for (int i = 0; i < a.getNValues(); i++)
    if (ga.getValueDirect(i) != 0.0) return false;
  return true;
}

int main() {
  run("Backward populates leaf grads", testBackwardPopulatesLeafGrads());
  run("Gradient accumulates", testGradientAccumulates());
  run("resetGradient zeroes", testResetGradient());
}
