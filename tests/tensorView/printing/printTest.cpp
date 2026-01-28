#include <mattTorch/mattTorch.h>
#include <iostream>

int main() {
  mattTorch::TensorView a({5, 5, 15, 15});
  a = 1.0;

  std::cout << a;
  return 0;

}
