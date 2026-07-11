
#include <mattTorch/function/multiplication/gradTransposeMatrix.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {

GradTransposeMatrix::GradTransposeMatrix(
    std::vector<Tensor> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions,
    bool transposeFirst)
    : savedTensors{std::move(savedTensors)},
      nextFunctions{std::move(nextFunctions)},
      transposeFirst{transposeFirst} {
}

void GradTransposeMatrix::backward(Tensor& inputGradient,
                                   bool higherDerivative) {
  assert(savedTensors.size() == 2);
  assert(nextFunctions.size() == 2);

  if (!higherDerivative) {
    savedTensors[0].setRequiresGrad(false);
    savedTensors[1].setRequiresGrad(false);
  }

  if (transposeFirst) {
    Tensor outputGradLHS =
        savedTensors[1].transposeMultiply(inputGradient, false);
    Tensor outputGradRHS = savedTensors[0].matrixMultiply(inputGradient);

    if (nextFunctions[0] != nullptr) {
      nextFunctions[0]->backward(outputGradLHS, higherDerivative);
    }
    if (nextFunctions[1] != nullptr) {
      nextFunctions[1]->backward(outputGradRHS, higherDerivative);
    }
  } else {
    Tensor outputGradLHS = inputGradient.matrixMultiply(savedTensors[1]);
    Tensor outputGradRHS =
        inputGradient.transposeMultiply(savedTensors[0], true);

    if (nextFunctions[0] != nullptr) {
      nextFunctions[0]->backward(outputGradLHS, higherDerivative);
    }
    if (nextFunctions[1] != nullptr) {
      nextFunctions[1]->backward(outputGradRHS, higherDerivative);
    }
  }
}
}  // namespace mattTorch::function
