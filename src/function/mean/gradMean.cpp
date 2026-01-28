

#include <mattTorch/function/mean/gradMean.h>

#include "mattTorch/tensor/tensorView/tensorView.h"

namespace mattTorch::function {

GradMean::GradMean(TensorView savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{savedTensor}, nextFunctions{nextFunctions} {
}

void GradMean::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradient({savedTensor.getDimensions()}, false);
  outputGradient = *inputGradient.getData() / savedTensor.getNValues();

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
