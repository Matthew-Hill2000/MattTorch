
#include "gradDivide.h"

GradDivide::GradDivide(std::vector<Tensor*> savedTensors,
                       std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradDivide::backward(Tensor& inputGradient) {}
