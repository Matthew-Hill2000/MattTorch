

#include <mattTorch/function/exponential/gradExponential.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {
GradExponential::GradExponential(
    TensorView savedTensor,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{savedTensor}, nextFunctions{nextFunctions} {
}

void GradExponential::backward(TensorView& inputGradient,
                               bool higherDerivative) {
  TensorView outputGrad = inputGradient * savedTensor.exponential();
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGrad, higherDerivative);
  }
}
}  // namespace mattTorch::function
