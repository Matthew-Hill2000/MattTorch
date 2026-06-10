#include <mattTorch/function/multiplication/gradMultiplyScalar.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {

GradMultiplyScalar::GradMultiplyScalar(
    double savedScalar,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalar{savedScalar}, nextFunctions{std::move(nextFunctions)} {}

void GradMultiplyScalar::backward(Tensor& inputGradient,
                                  bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGradient = inputGradient * savedScalar;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
