
#ifndef GRAD_DIVIDE_SCALAR_H
#define GRAD_DIVIDE_SCALAR_H

#include "../gradFunction.h"

class Tensor;

class GradDivideScalar : public GradFunction {
 private:
  std::vector<const Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradDivideScalar(std::vector<const Tensor*> savedTensors,
                   std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};

#endif
