#include <mattTorch/function/multiplication/gradMultiplyMatrix.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

GradMultiplyMatrix::GradMultiplyMatrix(
    std::vector<TensorView> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {
}

void GradMultiplyMatrix::backward(TensorView& inputGradient,
                                  bool higherDerivative) {
  TensorView outputGradLHS =
      inputGradient.transposeMultiply(savedTensors[1], false);
  TensorView outputGradRHS =
      savedTensors[0].transposeMultiply(inputGradient, true);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}
}  // namespace mattTorch::function
