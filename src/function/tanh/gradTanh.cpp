#include <mattTorch/function/tanh/gradTanh.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {

GradTanh::GradTanh(Tensor savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)} {
}

void GradTanh::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  if (!higherDerivative) {
    savedTensor.setRequiresGrad(false);
  }
  Tensor tanhOutput = savedTensor.tanh();
  Tensor outputGradient = inputGradient - inputGradient * tanhOutput * tanhOutput;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}


}  // namespace mattTorch::function
