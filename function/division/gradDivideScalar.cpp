#include "gradDivideScalar.h"

#include "../../tensor/tensor.h"

GradDivideScalar::GradDivideScalar(std::vector<const Tensor*> savedTensors,
                                   std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradDivideScalar::backward(Tensor& inputGradient) {}
