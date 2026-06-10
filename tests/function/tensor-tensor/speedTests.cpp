#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a * b;
  c.backward();
}

void benchDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a / b;
  c.backward();
}

void benchAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a + b;
  c.backward();
}

void benchSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a - b;
  c.backward();
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024});
  mattTorch::Tensor b = randomTensor({1024, 1024});

  run("Mul", [&] { benchMul(a, b); });
  run("Div", [&] { benchDiv(a, b); });
  run("Add", [&] { benchAdd(a, b); });
  run("Sub", [&] { benchSub(a, b); });
}
