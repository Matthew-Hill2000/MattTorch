#ifndef GRAD_DIVIDE_H
#define GRAD_DIVIDE_H

#include "../gradFunction.h"

class Tensor;

class GradDivide : public GradFunction {
 private:
  std::vector<const Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradDivide(std::vector<const Tensor*> savedTensors,
             std::vector<GradFunction*> nextFunction);
  void backward(Tensor& inputGradient) override;
};

#endif
