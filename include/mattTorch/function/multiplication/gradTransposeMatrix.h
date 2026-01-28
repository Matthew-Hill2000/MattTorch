#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

class GradTransposeMatrix : public GradFunction {
 private:
  // The values needed to calculate the gradient.
  std::vector<TensorView> savedTensors;
  // The GradFunction of the parent in the computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;
  bool transposeFirst;

 public:
  // Create a GradTransposeMatrix object with a vector containing the tensors
  // that were used to create it, as well as vector of their GradFunction
  // objects. These are used for calculation of the gradient and passing it
  // further up the computation graph
  GradTransposeMatrix(std::vector<TensorView> savedTensors,
                      std::vector<std::shared_ptr<GradFunction>> nextFunctions,
                      bool transposeFirst);

  // Calculate the gradient of this tensor with respect to each of the parents
  // as saved in savedTensors and multiply by the input gradient, which should
  // represent the gradient of the output with repsect to this tensor. Then,
  // pass each of these gradients to the respective GradFunction associated with
  // them.
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
