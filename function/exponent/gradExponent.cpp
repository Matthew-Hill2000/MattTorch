#include "gradExponent.h"

GradExponent::GradExponent(std::vector<Tensor*> savedTensors,
                           std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors} nextFunctions{nextFunctions} {}

void GradExponent::backward(Tensor& inputGradient) {
  TensorView outputGradient =
      savedTesors[0].get(data) * exponent * inputGradient.getData();
  for (int i{1}; i < exponent - 1; i++) {
    outputGradient *= outputGradient;
  }

  Tensor outGrad(outputGradient);
  nextFunctions[0]->backward(outGrad)
}
