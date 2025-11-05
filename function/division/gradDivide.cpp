
#include "gradDivide.h"

GradDivide::GradDivide(std::vector<Tensor*> savedTensors)
    : savedTensors{savedTensors} {}

void GradDivide::setNextFunctions(std::vector<GradFunction*> nextFunctions) {}

void GradDivide::backward(Tensor& inputGradient) {}
