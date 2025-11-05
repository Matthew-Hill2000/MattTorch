#include "../gradFunction.h"

class Tensor;

class GradAdd : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  std::vector<Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradAdd(std::vector<Tensor*> savedTensors);
  void backward(Tensor& inputGradient) override;
};
