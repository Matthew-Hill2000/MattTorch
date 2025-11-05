
#include "gradMultiplyScalar.h"

#include "../../tensor/tensor.h"

GradMultiplyScalar::GradMultiplyScalar(std::vector<Tensor*> savedTensors,
                                       std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradMultiplyScalar::backward(Tensor& inputGradient) {
  TensorView gradOneData = savedTensors[1]->getData() * inputGradient.getData();
  TensorView gradTwoData = savedTensors[0]->getData() * inputGradient.getData();

  Tensor gradOne(gradOneData);
  Tensor gradTwo(gradTwoData);

  nextFunctions[0]->backward(gradTwo);
  nextFunctions[1]->backward(gradOne);
}
