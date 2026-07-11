#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/tensor/tensor/tensor.h>

namespace mattTorch::function {

class GradReshape : public GradFunction {
 private:
  /// The input tensor from the forward pass, saved to determine the shape for
  /// gradient broadcasting
  Tensor savedTensor;

  /// The dimension along which the sum reduction was performed in the forward
  /// pass
  int reduceDim;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  GradReshape(Tensor savedTensor,
              std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
