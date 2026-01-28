
#pragma once

#include <memory>

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/mattTorch.h>

namespace mattTorch::function {
class GradReLU : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  TensorView backwardMask;
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  // Create a GradReLU object with a vector containing the tensors that were
  // used to create it, as well as vector of their GradFunction objects. These
  // are used for calculation of the gradient and passing it further up the
  // computation graph
  GradReLU(TensorView backwardMask,
          std::vector<std::shared_ptr<GradFunction>> nextFunctions);

// Calculate the gradient of this tensor with respect to each of the parents
// as saved in savedTensors and multiply by the input gradient, which should
// represent the gradient of the output with repsect to this tensor. Then,
// pass each of these gradients to the respective GradFunction associated with
// them.
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}
