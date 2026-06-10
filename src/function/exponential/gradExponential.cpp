
#include <mattTorch/function/exponential/gradExponential.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradExponential::GradExponential(
    Tensor savedTensor,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)} {
}

void GradExponential::backward(Tensor& inputGradient,
                               bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGrad = inputGradient * savedTensor.exponential();
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGrad, higherDerivative);
  }
}
}  // namespace mattTorch::function
