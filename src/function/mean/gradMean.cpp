

#include <mattTorch/function/mean/gradMean.h>

#include <cassert>
#include <utility>

#include "mattTorch/tensor/tensor/tensor.h"

namespace mattTorch::function {

GradMean::GradMean(Tensor savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)} {}

void GradMean::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGradient(savedTensor.getDimensions(), false);
  outputGradient = *inputGradient.getData() / savedTensor.getNValues();

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
