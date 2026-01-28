
#include <mattTorch/function/log/gradLog.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {
GradLog::GradLog(TensorView savedTensor,
                 std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{savedTensor}, nextFunctions{nextFunctions} {
}

void GradLog::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGrad = inputGradient / savedTensor;
  if (nextFunctions[0] != nullptr) {
  nextFunctions[0]->backward(outputGrad, higherDerivative);
  }
}
}  // namespace mattTorch::function
