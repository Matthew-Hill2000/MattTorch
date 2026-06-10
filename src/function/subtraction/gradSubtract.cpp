
#include <mattTorch/function/subtraction/gradSubtract.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradSubtract::GradSubtract(
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : nextFunctions{std::move(nextFunctions)} {}

void GradSubtract::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 2);

  Tensor outputGradLHS = inputGradient * 1.0;
  Tensor outputGradRHS = inputGradient * -1.0;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}
}  // namespace mattTorch::function
