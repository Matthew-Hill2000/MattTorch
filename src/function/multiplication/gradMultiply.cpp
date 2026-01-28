#include <mattTorch/function/multiplication/gradMultiply.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {
GradMultiply::GradMultiply(
    std::vector<TensorView> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradMultiply::backward(TensorView& inputGradient, bool higherDerivative) {
  if (!higherDerivative) {
    savedTensors[0].setRequiresGrad(false);
    savedTensors[1].setRequiresGrad(false);
  }
  TensorView outputGradLHS = savedTensors[1] * inputGradient;
  TensorView outputGradRHS = savedTensors[0] * inputGradient;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}
}  // namespace mattTorch::function
