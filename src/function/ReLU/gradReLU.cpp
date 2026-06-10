#include <mattTorch/function/ReLU/gradReLU.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {

GradReLU::GradReLU(Tensor backwardMask,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : backwardMask{std::move(backwardMask)},
      nextFunctions{std::move(nextFunctions)} {}

void GradReLU::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);
  assert(backwardMask.getDimensions() == inputGradient.getDimensions());

  backwardMask.setRequiresGrad(higherDerivative);

  Tensor outputGradient = backwardMask * inputGradient;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
