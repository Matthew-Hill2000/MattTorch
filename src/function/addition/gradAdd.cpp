#include <mattTorch/function/addition/gradAdd.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <memory>
#include <utility>
#include <vector>
namespace mattTorch::function {

GradAdd::GradAdd(std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : nextFunctions{std::move(nextFunctions)} {}

void GradAdd::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 2);

  Tensor outputGradLHS = inputGradient * 1.0;
  Tensor outputGradRHS = inputGradient * 1.0;
  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradLHS, higherDerivative);
  }
  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradRHS, higherDerivative);
  }
}

}  // namespace mattTorch::function
