#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <cstring>

namespace mattTorch::function {
GradAccumulator::GradAccumulator(std::shared_ptr<TensorView> gradient,
                                 const std::vector<int> dims)
    : gradient{gradient}, dims{dims} {}

void GradAccumulator::backward(TensorView& inputGradient,
                               bool higherDerivative) {
  if (gradient->getHasGrad() == false) {
    std::memcpy(gradient->getData(), inputGradient.getData(),
                gradient->getNValues() * sizeof(double));

    if (higherDerivative) {
      gradient->setGradFunction(inputGradient.getGradFunction());
      gradient->setRequiresGrad(true);
    }
    gradient->setHasGrad(true);
  } else {
    if (higherDerivative) {
      *gradient += inputGradient;
    } else {
      double* gradData = gradient->getData();
      double* inputData = inputGradient.getData();
      int nVals = gradient->getNValues();
      for (int i = 0; i < nVals; i++) {
        gradData[i] += inputData[i];
      }
    }
  }
}
void GradAccumulator::setGradient(std::shared_ptr<TensorView> newGrad) {
  gradient = newGrad;
}
}  // namespace mattTorch::function
