#include "gradMultiply.h"

#include "../../tensor/tensor.h"

GradMultiply::GradMultiply(std::vector<const Tensor*> savedTensors,
                           std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradMultiply::backward(Tensor& inputGradient) {
  TensorView gradOneData = savedTensors[1]->getData() * inputGradient.getData();
  TensorView gradTwoData = savedTensors[0]->getData() * inputGradient.getData();

  Tensor gradOne(gradOneData);
  Tensor gradTwo(gradTwoData);

  nextFunctions[0]->backward(gradTwo);
  nextFunctions[1]->backward(gradOne);
}
