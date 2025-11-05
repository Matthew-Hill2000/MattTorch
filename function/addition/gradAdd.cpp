#include "gradAdd.h"

GradAdd::GradAdd(std::vector<Tensor*> savedTensors,
                 std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradAdd::backward(Tensor& inputGradient) {
  nextFunctions[0]->backward(inputGradient);
  nextFunctions[1]->backward(inputGradient);
}
