
#include <mattTorch/function/broadcast/gradBroadcast.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

GradBroadcast::GradBroadcast(
    TensorView savedTensor, int broadcastPos,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{savedTensor},
      broadcastPos{broadcastPos},
      nextFunctions{nextFunctions} {
}

void GradBroadcast::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradient;
  outputGradient.setRequiresGrad(false);
  outputGradient = inputGradient.reductionSum(broadcastPos);

  if (nextFunctions[0] != nullptr) {
  nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
