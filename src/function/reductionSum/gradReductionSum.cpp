#include <mattTorch/function/reductionSum/gradReductionSum.h>

#include <cassert>
#include <utility>

#include "mattTorch/tensor/tensor/tensor.h"

namespace mattTorch::function {

GradReductionSum::GradReductionSum(
    std::vector<Tensor> savedTensors, int reduceDim,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{std::move(savedTensors)},
      reduceDim{reduceDim},
      nextFunctions{std::move(nextFunctions)} {}

void GradReductionSum::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(savedTensors.size() == 1);
  assert(nextFunctions.size() == 1);
  assert(reduceDim >= 0);
  assert(static_cast<std::size_t>(reduceDim) <
         savedTensors[0].getDimensions().size());

  int broadcastSize = savedTensors[0].getDimensions()[reduceDim];
  Tensor outputGradient = inputGradient.broadcast(reduceDim, broadcastSize);
  outputGradient.setRequiresGrad(false);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
