
#include <mattTorch/function/multiplication/gradTransposeMatrix.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

GradTransposeMatrix::GradTransposeMatrix(
    std::vector<TensorView> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions,
    bool transposeFirst)
    : savedTensors{savedTensors},
      nextFunctions{nextFunctions},
      transposeFirst{transposeFirst} {
}

void GradTransposeMatrix::backward(TensorView& inputGradient,
                                   bool higherDerivative) {
  if (transposeFirst) {
    TensorView outputGradLHS =
        savedTensors[1].transposeMultiply(inputGradient, false);
    TensorView outputGradRHS = savedTensors[0].matrixMultiply(inputGradient);

    if (nextFunctions[0] != nullptr) {
      nextFunctions[0]->backward(outputGradLHS, higherDerivative);
    }
    if (nextFunctions[1] != nullptr) {
      nextFunctions[1]->backward(outputGradRHS, higherDerivative);
    }
  } else {
    TensorView outputGradLHS = inputGradient.matrixMultiply(savedTensors[1]);
    TensorView outputGradRHS =
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
