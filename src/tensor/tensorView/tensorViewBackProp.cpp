#include <immintrin.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
#include <omp.h>

#include <cstring>
#include <memory>
namespace mattTorch {
// When called on a tensor it starts by calling the backward function of the
// gradFunction object associated with this tensor, which in turn iteratively
// calls the backwards functions of the parents of this tensor in the
// computational graph, iterativelly passing backwards the gradient and
// calculating succesive gradients of this tensor with respect to the tensors
// in the graph using the chain rule. The input gradient is used to
// initialise/seed the gradient and should be set to 1.0 by default
void TensorView::backward(TensorView& inputGradient, bool higherDerivative) {
  inputGradient.setRequiresGrad(higherDerivative);
  this->gradFunction->backward(inputGradient, higherDerivative);
}
// Adds the value of inputGradient to the value stored in this tensors
// gradStorage.
void TensorView::addGradient(TensorView& inputGradient) {
  if (!gradient) {
    gradient = std::make_shared<TensorView>(inputGradient.deepCopy());
  } else {
    *gradient += inputGradient;
  }
}

void TensorView::resetGradient() {
  double* gradData = getGradientData();
  memset(gradData, 0, nValues * sizeof(double));
}
}  // namespace mattTorch
