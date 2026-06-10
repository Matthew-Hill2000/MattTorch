#include <immintrin.h>
#include <mattTorch/tensor/tensor/tensor.h>
#include <omp.h>

#include <memory>

#include "mattTorch/function/accumulator/gradAccumulator.h"
namespace mattTorch {

void Tensor::backward(bool higherDerivative) {
  Tensor inputGradient(this->tensorData.dimensions);
  inputGradient = 1.0;
  backward(inputGradient, higherDerivative);
}

void Tensor::backward(Tensor& inputGradient, bool higherDerivative) {
  inputGradient.setRequiresGrad(higherDerivative);
  this->gradFunction->backward(inputGradient, higherDerivative);
}

void Tensor::resetGradient() {
  gradient = std::make_shared<Tensor>(tensorData.dimensions, false);
  gradient->setRequiresGrad(false);
  std::static_pointer_cast<function::GradAccumulator>(gradFunction)
      ->setGradient(gradient);
}
}  // namespace mattTorch
