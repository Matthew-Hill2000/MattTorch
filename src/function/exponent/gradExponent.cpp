#include <mattTorch/function/exponent/gradExponent.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradExponent::GradExponent(
    std::vector<int> savedScalars, std::vector<Tensor> savedTensors,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalars{std::move(savedScalars)},
      savedTensors{std::move(savedTensors)},
      nextFunctions{std::move(nextFunctions)} {
}

void GradExponent::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(savedScalars.size() == 1);
  assert(savedTensors.size() == 1);
  assert(nextFunctions.size() == 1);

  Tensor outputGradient = savedScalars[0] * inputGradient;
  for (int i{0}; i < savedScalars[0] - 1; i++) {
    outputGradient *= savedTensors[0];
  }
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
