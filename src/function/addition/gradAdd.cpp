#include <mattTorch/function/addition/gradAdd.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <memory>
#include <vector>
namespace mattTorch::function {
GradAdd::GradAdd(std::vector<TensorView> savedTensors,
                 std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradAdd::backward(TensorView& inputGradient, bool higherDerivative) {
  TensorView outputGradLHS = inputGradient * 1.0;
  TensorView outputGradRHS = inputGradient * 1.0;
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}
}  // namespace mattTorch::function
