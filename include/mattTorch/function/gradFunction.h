#pragma once
#include <memory>
#include <vector>

namespace mattTorch {

class TensorView;

class GradFunction {
 private:
  // Stores pointers to the GradFunction objects of the parents in the graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  // Calculate the gradient of this tensor with respect to each of the parents
  // as saved in savedTensors and multiply by the input gradient, which should
  // represent the gradient of the output with repsect to this tensor. Then,
  // pass each of these gradients to the respective GradFunction associated with
  // them.
  virtual void backward(TensorView& inputGradient, bool higherDerivative) = 0;
  virtual ~GradFunction() = default;
};

}  // namespace mattTorch
