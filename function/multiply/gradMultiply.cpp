#include "gradMultiply.h"

GradMultiply::GradMultiply(std::vector<Tensor*> savedTensors,
                           std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradMultiply::backward(Tensor& inputGradient) {
  TensorView gradOne = savedTensors[1]->getData() * inputGradient->getData();
  TensorView gradTwo = savedTensors[0]->getData() * inputGradient->getData();

  nextFunctions[0]->backward(gradTwo);
  nextFunctions[1]->backward(gradOne);
}
