#include <mattTorch/function/division/gradDivideScalar.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cassert>
#include <memory>
#include <utility>

namespace mattTorch::function {
GradDivideScalar::GradDivideScalar(
    double savedScalar, std::shared_ptr<Tensor> savedTensor,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions, bool numerator)
    : savedScalar{savedScalar},
      savedTensor{std::move(savedTensor)},
      nextFunctions{std::move(nextFunctions)},
      numerator{numerator} {
}

void GradDivideScalar::backward(Tensor& inputGradient,
                                bool higherDerivative) {
  assert(nextFunctions.size() == 1);

  if (numerator) {
    Tensor outputGradient = inputGradient / savedScalar;

    if (nextFunctions[0] != nullptr) {
      this->nextFunctions[0]->backward(outputGradient, higherDerivative);
    }
  } else {
    Tensor outputGradient =
        (inputGradient * savedScalar * -1.0) / (*savedTensor * *savedTensor);

    if (nextFunctions[0] != nullptr) {
      this->nextFunctions[0]->backward(outputGradient, higherDerivative);
    }
  }
}
}  // namespace mattTorch::function
