
#include <mattTorch/function/log/gradLog.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradLog::GradLog(Tensor savedTensor,
                 std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)} {
}

void GradLog::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  if (!higherDerivative) {
    savedTensor.setRequiresGrad(false);
  }

  Tensor outputGrad = inputGradient / savedTensor;
  if (nextFunctions[0] != nullptr) {
  nextFunctions[0]->backward(outputGrad, higherDerivative);
  }
}
}  // namespace mattTorch::function
