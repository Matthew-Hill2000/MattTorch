#include <mattTorch/function/reductionSum/gradReductionSum.h>

#include "mattTorch/tensor/tensorView/tensorView.h"

namespace mattTorch::function {

GradReductionSum::GradReductionSum(
    std::vector<TensorView> savedTensors, int reduceDim,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{savedTensors},
      reduceDim{reduceDim},
      nextFunctions{nextFunctions} {
}

void GradReductionSum::backward(TensorView& inputGradient, bool higherDerivative) {
  // Broadcast back along the reduced dimension
  int broadcastSize = savedTensors[0].getDimensions()[reduceDim];
  TensorView outputGradient = inputGradient.broadcast(reduceDim, broadcastSize);
  outputGradient.setRequiresGrad(false);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
