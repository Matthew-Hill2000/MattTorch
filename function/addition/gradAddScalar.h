
#ifndef GRAD_ADD_SCALAR_H
#define GRAD_ADD_SCALAR_H

#include "../gradFunction.h"

class Tensor;

class GradAddScalar : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  std::vector<Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradAddScalar(std::vector<Tensor*> savedTensors,
                std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};

#endif
