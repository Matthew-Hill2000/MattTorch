#include "gradExponent.h"

GradExponent::GradExponent(std::vector<Tensor*> savedTensors,
                           std::vector<GradFunction*> nextFunctions,
                           int exponent)
    : savedTensors{savedTensors},
      nextFunctions{nextFunctions},
      exponent{exponent} {}

void GradExponent::backward(Tensor& inputGradient) {
  TensorView outputGradient =
      savedTensors[0]->getData() * exponent * inputGradient.getData();
  for (int i{1}; i < exponent - 1; i++) {
    outputGradient *= outputGradient;
  }

  Tensor outGrad(outputGradient);
  nextFunctions[0]->backward(outGrad);
}
