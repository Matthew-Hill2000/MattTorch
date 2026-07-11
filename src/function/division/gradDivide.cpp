#include <mattTorch/function/division/gradDivide.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <utility>

namespace mattTorch::function {
GradDivide::GradDivide(std::vector<Tensor> savedTensors,
                       std::vector<std::shared_ptr<GradFunction>> nextFunctions)
    : savedTensors{std::move(savedTensors)},
      nextFunctions{std::move(nextFunctions)} {
}

void GradDivide::backward(Tensor& inputGradient, bool higherDerivative) {
  assert(savedTensors.size() == 2);
  assert(nextFunctions.size() == 2);

  if (!higherDerivative) {
    savedTensors[0].setRequiresGrad(false);
    savedTensors[1].setRequiresGrad(false);
  }

  Tensor outputGradNumerator = inputGradient / savedTensors[1];
  Tensor outputGradDenominator = (inputGradient * savedTensors[0] * -1.0) /
                                     (savedTensors[1] * savedTensors[1]);

  if (nextFunctions[0] != nullptr) {
    nextFunctions[0]->backward(outputGradNumerator, higherDerivative);
  }

  if (nextFunctions[1] != nullptr) {
    nextFunctions[1]->backward(outputGradDenominator, higherDerivative);
  }
}
}  // namespace mattTorch::function
