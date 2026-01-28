

#include <mattTorch/function/subtraction/gradSubtractScalar.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch::function {
GradSubtractScalar::GradSubtractScalar(
    double savedScalar,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalar{savedScalar}, nextFunctions{nextFunctions} {}

void GradSubtractScalar::backward(TensorView& inputGradient,
                                  bool higherDerivative) {
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(inputGradient, higherDerivative);
  }
}
}  // namespace mattTorch::function
