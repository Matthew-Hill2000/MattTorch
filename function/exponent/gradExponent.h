#ifndef GRAD_EXPONENT_H
#define GRAD_EXPONENT_H

#include "../../tensor/tensor.h"
#include "../gradFunction.h"

class GradExponent : public GradFunction {
 private:
  std::vector<const Tensor*> savedTensors;
  int exponent;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradExponent(std::vector<const Tensor*> savedTensors,
               std::vector<GradFunction*> nextFunctions, int exponent);
  void backward(Tensor& inputGradient) override;
};

#endif
