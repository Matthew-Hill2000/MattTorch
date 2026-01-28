#pragma once

#include <mattTorch/function/gradFunction.h>


namespace mattTorch::function {
class GradDivideScalar : public GradFunction {
 private:
  // The values needed to calculate the gradient.
  double savedScalar;
  std::shared_ptr<TensorView> savedTensor;
  // The GradFunction of the parent in the computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;
  bool numerator;

 public:
  // Create a GradDivideScalar object with a vector containing the scalar that
  // was used to create it, as well as vector of shared pointers to the
  // GradFunction object of the parent tensor. These are used for calculation of
  // the gradient and passing it further up the computation graph
  GradDivideScalar(double savedScalar, std::shared_ptr<TensorView> savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions,
                   bool numerator = true);

  // Calculate the gradient of this tensor with respect to its tensor parent and
  // multiply by the input gradient, which should represent the gradient of the
  // output with repsect to this tensor. Then, pass the gradients to the
  // GradFunction associated with the tensor parent.
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
