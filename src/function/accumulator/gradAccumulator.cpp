#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cstring>

namespace mattTorch::function {
GradAccumulator::GradAccumulator(std::shared_ptr<Tensor> gradient, Dims dims)
    : gradient{gradient}, dims{dims} {}

void GradAccumulator::backward(Tensor& inputGradient, bool higherDerivative) {
  if (!gradient->getHasGrad()) {
    std::memcpy(gradient->getData(), inputGradient.getData(),
                gradient->getNValues() * sizeof(double));

    if (higherDerivative) {
      gradient->setGradFunction(inputGradient.getGradFunction());
      gradient->setRequiresGrad(true);
    }

    gradient->setHasGrad(true);
    return;
  }

  *gradient += inputGradient;
}

void GradAccumulator::setGradient(std::shared_ptr<Tensor> newGrad) {
  gradient = newGrad;
}
}  // namespace mattTorch::function
