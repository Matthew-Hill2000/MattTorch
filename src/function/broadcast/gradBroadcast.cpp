
#include <mattTorch/function/broadcast/gradBroadcast.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {

GradBroadcast::GradBroadcast(
    Tensor savedTensor, int broadcastPos,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      broadcastPos{broadcastPos},
      nextFunctions{std::move(nextFunctions)} {}

void GradBroadcast::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGradient;
  outputGradient.setRequiresGrad(false);
  outputGradient = inputGradient.reductionSum(broadcastPos);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
