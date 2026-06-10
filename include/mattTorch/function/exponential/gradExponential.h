#pragma once

#include <mattTorch/function/gradFunction.h>

#include "mattTorch/tensor/tensor/tensor.h"

namespace mattTorch::function {

/**
 * @brief Gradient function for the natural exponential function.
 *
 * The GradExponential class represents the natural exponential function in the
 * computational graph, computing gradients for the elementwise exponential
 * operation. For the operation @f$C = e^A@f$ where @f$A@f$ is a tensor, the
 * gradient computation follows from the derivative of the exponential
 * function:
 * @f[
 *     \frac{\partial C}{\partial A} = e^A
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with the exponential of the input according
 * to the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \odot e^A
 * @f]
 *
 * This GradFunction saves the output tensor @f$e^A@f$ from the forward pass
 * rather than the input tensor @f$A@f$, as the output is exactly what is
 * needed for the gradient computation. This avoids recomputing the exponential
 * during the backward pass, improving computational efficiency.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradLog for the gradient function for the natural logarithm.
 * @see GradExponent for the gradient function for integer power operations.
 */
class GradExponential : public GradFunction {
 private:
  /// The output tensor from the forward pass (e^A), saved for use in gradient
  /// computation to avoid recomputing the exponential
  Tensor savedTensor;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradExponential gradient function
   *
   * Creates a GradExponential object that stores the output tensor from the
   * forward pass and a pointer to the parent tensor's GradFunction for
   * gradient propagation during backpropagation.
   *
   * @param savedTensor The output tensor (e^A) from the forward pass, saved
   * for use in gradient computation
   * @param nextFunction A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradExponential(Tensor savedTensor,
                  std::vector<std::shared_ptr<GradFunction>> nextFunction);

  /**
   * @brief Computes and propagates gradients for the exponential operation
   *
   * Implements the backward pass for the natural exponential function by
   * multiplying the incoming gradient elementwise with the saved output tensor
   * (which contains e^A). Since the derivative of e^x is e^x, and the output
   * was already computed during the forward pass, this computation is
   * efficient. The method calls backward() on the parent tensor's GradFunction
   * to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensor to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the exponential operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
