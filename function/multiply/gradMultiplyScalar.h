
#ifndef GRAD_MULTIPLY_SCALAR_H
#define GRAD_MULTIPLY_SCALAR_H

#include "../gradFunction.h"

class Tensor;

class GradMultiplyScalar : public GradFunction {
 private:
  std::vector<const Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradMultiplyScalar(std::vector<const Tensor*> savedTensors,
                     std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};

#endif
