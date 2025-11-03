#include "../../tensor/tensor.h"
#include "../function.h"

class multiplyFunction : public Function {
 public:
  Tensor forward(Tensor a, Tensor b) override;
  void backward() override;
};
