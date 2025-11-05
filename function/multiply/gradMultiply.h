#include "../gradFunction.h"

class Tensor;

class GradMultiply : public GradFunction {
 private:
  std::vector<Tensor*> savedTensors;

 public:
  GradMultiply(std::vector<Tensor*> savedTensors,
               std::vector<GradFunction*> nextFunctions);
  void backward(Tensor& inputGradient) override;
};
