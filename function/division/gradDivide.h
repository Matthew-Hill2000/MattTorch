#include "../gradFunction.h"

class Tensor;

class GradDivide : public GradFunction {
 private:
  // The tensors needed to calculate the gradient.
  std::vector<Tensor*> savedTensors;

 public:
  GradDivide(std::vector<Tensor*> savedTensors);
  void backward(Tensor& inputGradient) override;
  void setNextFunctions(std::vector<GradFunction*> nextFunctions) override;
};
