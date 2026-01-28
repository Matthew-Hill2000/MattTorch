#pragma once

#include <mattTorch/function/gradFunction.h>

#include "mattTorch/tensor/tensorView/tensorView.h"

namespace mattTorch::function {
class GradAccumulator : public GradFunction {
 private:
  // A pointer to the tensor that this GradAccumulator belongs to.
  std::shared_ptr<TensorView> gradient;
  const std::vector<int> dims;

 public:
  // Construct a GraddAccumulator with a pointer that points to the tensor that
  // owns it.
  GradAccumulator(std::shared_ptr<TensorView> gradient,
                  const std::vector<int> dims);

  // Calculate the gradient of this tensor with respect to each of the children
  // as saved in savedTensors and multiply by the input gradient, which should
  // represent the gradient of the output with repsect to this tensor. Then,
  // pass each of these gradients to the respective GradFunction associated with
  // them.
  void backward(TensorView& inputGradient, bool higherDerivative) override;
  void setGradient(std::shared_ptr<TensorView> newGrad);
};
}  // namespace mattTorch::function
