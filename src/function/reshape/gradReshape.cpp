#include <mattTorch/function/reshape/gradReshape.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>

namespace mattTorch::function {

GradReshape::GradReshape(
    Tensor savedTensor,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)} {}

void GradReshape::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGradient;
  outputGradient.setRequiresGrad(false);

  outputGradient = inputGradient.reshape(savedTensor.getDimensions());

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
