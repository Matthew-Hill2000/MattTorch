
#include "gradAddScalar.h"

GradAddScalar::GradAddScalar(std::vector<const Tensor*> savedTensors,
                             std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradAddScalar::backward(Tensor& inputGradient) {
  nextFunctions[0]->backward(inputGradient);
  nextFunctions[1]->backward(inputGradient);
}
