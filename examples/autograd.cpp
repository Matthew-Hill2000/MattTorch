#include <mattTorch/mattTorch.h>

#include <iostream>

int main() {
  mattTorch::TensorView x(std::vector<int>{10});

  for (int i{0}; i < x.getNValues(); i++) {
    x.setValueDirect(i, i);
  }

  mattTorch::TensorView y =  2*x*x*x ;

  std::cout << x << std::endl;
  std::cout << y << std::endl;

  mattTorch::TensorView inputGrad(y.getDimensions());
  inputGrad = 1.0;
  y.backward(inputGrad, true);

  mattTorch::TensorView xGrad1 = x.detachGradient();
  std::cout << xGrad1 << std::endl;

  xGrad1.backward(inputGrad, true);
  mattTorch::TensorView xGrad2 = x.detachGradient();
  std::cout << xGrad2 << std::endl;

  xGrad2.backward(inputGrad);
  mattTorch::TensorView xGrad3 = x.detachGradient();
  std::cout << xGrad3 << std::endl;

  return 1;
}
