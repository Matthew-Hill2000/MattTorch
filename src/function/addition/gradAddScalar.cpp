
#include <mattTorch/function/addition/gradAddScalar.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch::function {
GradAddScalar::GradAddScalar(
    double savedScalar,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedScalar{savedScalar}, nextFunctions{nextFunctions} {}

void GradAddScalar::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGrad = inputGradient * 1.0;
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGrad, higherDerivative);
  }
}
}  // namespace mattTorch::function
