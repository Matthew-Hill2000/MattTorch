
#include <iostream>
#include <vector>

#include "mattTorch/tensor/tensorView/tensorView.h"

void vectorPrint(const std::vector<int>& vector) {
  for (auto element : vector) {
    std::cout << element << ", ";
  }
  std::cout << std::endl;
}

int main() {
  mattTorch::TensorView a({4, 4});
  mattTorch::TensorView b({4, 4});

  a = 2.0;
  b = 3.0;

  mattTorch::TensorView c = a / (b * b);
  std::cout << c << std ::endl;

  mattTorch::TensorView gradSeed(c.getDimensions());
  gradSeed = 1.0;
  c.backward(gradSeed);
  std::cout << a.detachGradient() << std::endl;
  std::cout << b.detachGradient() << std::endl;
}
