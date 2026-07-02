
#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/tensor/tensor/tensor.h>

namespace mattTorch::function {

class GradSum : public GradFunction {
 private:
  Tensor savedTensor;

  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  GradSum(Tensor savedTensor,
          std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
