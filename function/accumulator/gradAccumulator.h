#ifndef GRAD_ACCUMULATOR_H
#define GRAD_ACCUMULATOR_H

#include "../../tensor/tensor.h"
#include "../gradFunction.h"

class GradAccumulator : public GradFunction {
 private:
  Tensor* savedTensor;

 public:
  GradAccumulator(Tensor* savedTensor);
  void backward(Tensor& inputGradient) override;
};

#endif
