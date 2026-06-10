#pragma once

#include "mattTorch/tensor/tensor/tensor.h"
#include <mattTorch/function/gradFunction.h>


namespace mattTorch::function {

/**
 * @brief Gradient function for the natural logarithm.
 *
 * The GradLog class represents the natural logarithm function in the
 * computational graph, computing gradients for the elementwise logarithm
 * operation. For the operation @f$C = \ln(A)@f$ where @f$A@f$ is a tensor, the
 * gradient computation follows from the derivative of the logarithm:
 * @f[
 *     \frac{\partial C}{\partial A} = \frac{1}{A}
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with the reciprocal of the input according
 * to the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \oslash A
 * @f]
 *
 * This GradFunction saves the input tensor from the forward pass as it is
 * required to compute the gradient using the formula @f$1/A@f$ during the
 * backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradExponential for the gradient function for the natural exponential
 * function.
 */
class GradLog : public GradFunction {
private:
  /// The input tensor from the forward pass, required for computing the
  /// gradient as 1/A
  Tensor savedTensor;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

public:
  /**
   * @brief Constructs a GradLog gradient function
   *
   * Creates a GradLog object that stores the input tensor from the forward
   * pass and a pointer to the parent tensor's GradFunction for gradient
   * propagation during backpropagation.
   *
   * @param savedTensor The input tensor from the forward pass, required for
   * computing the gradient as 1/A
   * @param nextFunction A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradLog(Tensor savedTensor,
             std::vector<std::shared_ptr<GradFunction>> nextFunction);

  /**
   * @brief Computes and propagates gradients for the logarithm operation
   *
   * Implements the backward pass for the natural logarithm by computing the
   * reciprocal of the saved input tensor and multiplying it elementwise with
   * the incoming gradient. Since the derivative of ln(x) is 1/x, the gradient
   * is computed by dividing the incoming gradient by the input tensor. The
   * method calls backward() on the parent tensor's GradFunction to continue
   * the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensor to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the logarithm operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor &inputGradient, bool higherDerivative) override;
};
}
