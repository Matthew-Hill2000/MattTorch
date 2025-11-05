#ifndef GRAD_SUBTRACT_H
#define GRAD_SUBTRACT_H

#include "../gradFunction.h"

class Tensor;

class GradSubtract : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  std::vector<const Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradSubtract(std::vector<const Tensor*> savedTensors,
               std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};

#endif
