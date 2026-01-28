#include <mattTorch/function/multiplication/gradMultiplyScalar.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

GradMultiplyScalar::GradMultiplyScalar(
    double savedScalar,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalar{savedScalar}, nextFunctions{nextFunctions} {}

void GradMultiplyScalar::backward(TensorView& inputGradient,
                                  bool higherDerivative) {
  TensorView outputGradient = inputGradient * savedScalar;

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
