#include <mattTorch/function/division/gradDivideScalar.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <memory>

namespace mattTorch::function {
GradDivideScalar::GradDivideScalar(
    double savedScalar, std::shared_ptr<TensorView> savedTensor,
    std::vector<std::shared_ptr<GradFunction>> nextFunctions, bool numerator)
    : savedScalar{savedScalar},
      savedTensor{savedTensor},
      nextFunctions{nextFunctions},
      numerator{numerator} {
}

void GradDivideScalar::backward(TensorView& inputGradient,
                                bool higherDerivative) {
  if (numerator) {
    TensorView outputGradient = inputGradient / savedScalar;

    if (nextFunctions[0] != nullptr) {
      this->nextFunctions[0]->backward(outputGradient, higherDerivative);
    }
  } else {
    TensorView outputGradient =
        (inputGradient * savedScalar * -1.0) / (*savedTensor * *savedTensor);

    if (nextFunctions[0] != nullptr) {
      this->nextFunctions[0]->backward(outputGradient, higherDerivative);
    }
  }
}
}  // namespace mattTorch::function
