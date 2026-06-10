#include <mattTorch/function/multiplication/gradMultiplyMatrix.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {

GradMultiplyMatrix::GradMultiplyMatrix(
    std::vector<Tensor> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{std::move(savedTensors)},
      nextFunctions{std::move(nextFunctions)} {}

void GradMultiplyMatrix::backward(Tensor& inputGradient,
                                  bool higherDerivative) {
  assert(savedTensors.size() == 2);
  assert(nextFunctions.size() == 2);

  if (!higherDerivative) {
    savedTensors[0].setRequiresGrad(false);
    savedTensors[1].setRequiresGrad(false);
  }
  Tensor outputGradLHS =
      inputGradient.transposeMultiply(savedTensors[1], false);
  Tensor outputGradRHS =
      savedTensors[0].transposeMultiply(inputGradient, true);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}
}  // namespace mattTorch::function
