#include "gradAccumulator.h"

GradAccumulator::GradAccumulator(Tensor* savedTensor)
    : savedTensor{savedTensor} {}

void GradAccumulator::backward(Tensor& inputGradient) {
  this->savedTensor->addGradient(inputGradient);
}
