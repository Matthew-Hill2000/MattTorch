#include <mattTorch/function/tanh/gradTanh.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

GradTanh::GradTanh(TensorView savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{savedTensor}, nextFunctions{nextFunctions} {
}

void GradTanh::backward(TensorView& inputGradient, bool higherDerivative) {
  if (!higherDerivative) {
    savedTensor.setRequiresGrad(false);
  }
  TensorView tanhOutput = savedTensor.tanh();
  TensorView outputGradient = inputGradient - inputGradient * tanhOutput * tanhOutput;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}


}  // namespace mattTorch::function
