#include <vector>

class Tensor;

class GradFunction {
 private:
  std::vector<GradFunction*> nextFunctions;

 public:
  virtual void backward(Tensor& inputGradient) = 0;
  virtual ~GradFunction() = default;
};
