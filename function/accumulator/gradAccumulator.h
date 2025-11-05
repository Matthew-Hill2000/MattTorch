
#include "../../tensor/tensor.h"
#include "../gradFunction.h"

class GradAccumulator : public GradFunction {
 public:
  GradAccumulator(Tensor* savedTensor);
  void backward(Tensor& inputGradient) override;
};
