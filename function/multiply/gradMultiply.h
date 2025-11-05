#ifndef GRAD_MULTIPLY_H
#define GRAD_MULTIPLY_H

#include "../gradFunction.h"

class Tensor;

class GradMultiply : public GradFunction {
 private:
  std::vector<const Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradMultiply(std::vector<const Tensor*> savedTensors,
               std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};

#endif
