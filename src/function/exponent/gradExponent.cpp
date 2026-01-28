#include <mattTorch/function/exponent/gradExponent.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {
GradExponent::GradExponent(
    std::vector<int> savedScalars, std::vector<TensorView> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalars{savedScalars},
      savedTensors{savedTensors},
      nextFunctions{nextFunctions} {
}

void GradExponent::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradient = savedScalars[0] * inputGradient;
  for (int i{0}; i < savedScalars[0] - 1; i++) {
    outputGradient *= savedTensors[0];
  }
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
