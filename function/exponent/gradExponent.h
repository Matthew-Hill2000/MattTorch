#include "../../tensor/tensor.h"
#include "../gradFunction.h"

class GradExponent : public GradFunction {
 private:
  std::vector<Tensor*> savedTensors;
  int exponent;

 public:
  GradExponent(std::vector<Tensor*> savedTensors, int exponent);
  void backward(Tensor& inputGradient) override;
};
