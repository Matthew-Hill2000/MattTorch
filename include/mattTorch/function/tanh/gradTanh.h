#pragma once

#include "mattTorch/tensor/tensorView/tensorView.h"
#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

class GradTanh : public GradFunction {
 private:
  // The output of tanh (needed to compute gradient: 1 - tanh(x)^2)
  TensorView savedTensor;
  // The GradFunction of the parent in the computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  // Create a GradTanh object with the output of the tanh operation
  // and the GradFunction of the input tensor
  GradTanh(TensorView tanhOutput,
           std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  // Calculate the gradient: d(tanh(x))/dx = 1 - tanh(x)^2
  // Multiply by input gradient and pass to the next function
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};

}  // namespace mattTorch::function
