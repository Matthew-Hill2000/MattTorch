
#include <mattTorch/function/addition/gradAddScalar.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradAddScalar::GradAddScalar(
    double savedScalar,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalar{savedScalar}, nextFunctions{std::move(nextFunctions)} {}

void GradAddScalar::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGrad = inputGradient * 1.0;
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGrad, higherDerivative);
  }
}
}  // namespace mattTorch::function
