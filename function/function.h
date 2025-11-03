#include "../tensor/tensor.h"

class Function {
 private:
  Tensor* child;
  std::vector<Tensor*> parents;

 public:
  virtual Tensor forward(Tensor a, Tensor b) = 0;
  virtual void backward() = 0;
  virtual ~Function() = default;
};
