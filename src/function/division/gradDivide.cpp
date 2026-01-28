#include <mattTorch/function/division/gradDivide.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {
GradDivide::GradDivide(std::vector<TensorView> savedTensors,
                       std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {
}

void GradDivide::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradNumerator = inputGradient / savedTensors[1];
  TensorView outputGradDenominator = (inputGradient * savedTensors[0] * -1.0) /
                                     (savedTensors[1] * savedTensors[1]);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradNumerator, higherDerivative);
  }

  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradDenominator, higherDerivative);
  }
}
}  // namespace mattTorch::function
