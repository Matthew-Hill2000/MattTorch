#ifndef GRAD_SUBTRACT_SCALAR_H
#define GRAD_SUBTRACT_SCALAR_H

#include "../gradFunction.h"

class Tensor;

class GradSubtractScalar : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  std::vector<const Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradSubtractScalar(std::vector<const Tensor*> savedTensors,
                     std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};

#endif
