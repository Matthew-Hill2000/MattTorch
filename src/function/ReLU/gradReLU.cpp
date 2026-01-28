#include <mattTorch/function/ReLU/gradReLU.h>

namespace mattTorch::function {

GradReLU::GradReLU(TensorView backwardMask,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : backwardMask{backwardMask}, nextFunctions{nextFunctions} {
}

void GradReLU::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradient = backwardMask * inputGradient;

  if (nextFunctions[0] != nullptr) {
  nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
