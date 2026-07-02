#include <mattTorch/function/sum/gradSum.h>

#include <cassert>
#include <utility>

#include "mattTorch/tensor/tensor/tensor.h"

namespace mattTorch::function {

GradSum::GradSum(Tensor savedTensor,
                 std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)} {}

void GradSum::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  Tensor outputGradient(savedTensor.getDimensions(), false);
  outputGradient = *inputGradient.getData();

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradient, higherDerivative);
  }
}

}  // namespace mattTorch::function
