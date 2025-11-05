
#include "gradSubtract.h"

GradSubtract::GradSubtract(std::vector<Tensor*> savedTensors,
                           std::vector<GradFunction*> nextFunctions)
    : savedTensors{savedTensors}, nextFunctions{nextFunctions} {}

void GradSubtract::backward(Tensor& inputGradient) {
  TensorView negativeGradient = -1 * inputGradient.getData();
  Tensor negGrad(negativeGradient);

  nextFunctions[0]->backward(inputGradient);
  nextFunctions[1]->backward(negGrad);
}
