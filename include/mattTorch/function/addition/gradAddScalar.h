#pragma once

#include <mattTorch/function/gradFunction.h>


namespace mattTorch::function {
class GradAddScalar : public GradFunction {
 private:
  // The values needed to calculate the gradient.
  double savedScalar;
  // The GradFunction of the parent in the computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  // Create a GradAddScalar object with a vector containing the scalar that
  // was used to create it, as well as vector of shared pointers to the
  // GradFunction object of the parent tensor. These are used for calculation of
  // the gradient and passing it further up the computation graph
  GradAddScalar(double savedScalar,
                std::vector<std::shared_ptr<GradFunction>> nextFunctions);
  //
  // Calculate the gradient of this tensor with respect to the parent
  // and multiply by the input gradient, which should
  // represent the gradient of the output with repsect to this tensor. Then,
  // pass the gradient to the GradFunction of the parent.
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}
