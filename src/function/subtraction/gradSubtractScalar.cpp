

#include <mattTorch/function/subtraction/gradSubtractScalar.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradSubtractScalar::GradSubtractScalar(
    double savedScalar,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalar{savedScalar}, nextFunctions{std::move(nextFunctions)} {}

void GradSubtractScalar::backward(Tensor& inputGradient,
                                  bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(inputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
