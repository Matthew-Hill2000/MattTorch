

#include "gradSubtractScalar.h"

#include "../../tensor/tensor.h"

GradSubtractScalar::GradSubtractScalar(std::vector<Tensor*> savedTensors,
                                       std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradSubtractScalar::backward(Tensor& inputGradient) {
  TensorView negativeGradient = inputGradient.getData() * -1;
  Tensor negGrad(negativeGradient);

  nextFunctions[0]->backward(inputGradient);
  nextFunctions[1]->backward(negGrad);
}
