#include <immintrin.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
#include <omp.h>

#include <memory>

#include "mattTorch/function/accumulator/gradAccumulator.h"
namespace mattTorch {

void TensorView::backward(bool higherDerivative) {
  TensorView inputGradient(this->dimensions);
  inputGradient = 1.0;
  backward(inputGradient, higherDerivative);
}

void TensorView::backward(TensorView& inputGradient, bool higherDerivative) {
  inputGradient.setRequiresGrad(higherDerivative);
  this->gradFunction->backward(inputGradient, higherDerivative);
}

void TensorView::addGradient(TensorView& inputGradient) {
  if (!gradient) {
    gradient = std::make_shared<TensorView>(inputGradient.deepCopy());
  } else {
    *gradient += inputGradient;
  }
}

void TensorView::resetGradient() {
  gradient = std::make_shared<TensorView>(dimensions, false);
  gradient->setRequiresGrad(false);
  std::static_pointer_cast<function::GradAccumulator>(gradFunction)
      ->setGradient(gradient);
}
}  // namespace mattTorch
