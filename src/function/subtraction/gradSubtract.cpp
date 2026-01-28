
#include <mattTorch/function/subtraction/gradSubtract.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch::function {
GradSubtract::GradSubtract(
    std::vector<TensorView> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {
}

void GradSubtract::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradLHS = inputGradient * 1.0;
  TensorView outputGradRHS = inputGradient * -1.0;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}
}  // namespace mattTorch::function
