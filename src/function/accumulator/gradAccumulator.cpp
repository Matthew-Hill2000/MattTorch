#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensorView/tensorView.h>



namespace mattTorch::function {
GradAccumulator::GradAccumulator(std::shared_ptr<TensorView> gradient,
                                 const std::vector<int> dims)
    : gradient{gradient}, dims{dims} {}

void GradAccumulator::backward(TensorView& inputGradient,
                               bool higherDerivative) {
  if (gradient->getHasGrad() == false) {
    *gradient = inputGradient;
  } else {
    *gradient += inputGradient;
  }
}
void GradAccumulator::setGradient(std::shared_ptr<TensorView> newGrad) {
  gradient = newGrad;
}
}  // namespace mattTorch::function
