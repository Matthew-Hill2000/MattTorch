#include "../gradFunction.h"

class Tensor;

class GradSubtract : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  std::vector<Tensor*> savedTensors;
  std::vector<GradFunction*> nextFunctions;

 public:
  GradSubtract(std::vector<Tensor*> savedTensors);
  void backward(Tensor& inputGradient) override;
};
