#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for tensor-scalar subtraction.
 *
 * The GradSubtractScalar class represents the subtraction of a scalar value
 * from a tensor in the computational graph, computing gradients for operations
 * of the form @f$C = A - s@f$ where @f$A@f$ is a tensor and @f$s@f$ is a
 * scalar value. The gradient computation follows from the derivative:
 * @f[
 *     \frac{\partial C}{\partial A} = 1
 * @f]
 *
 * The scalar value does not participate in gradient computation as it is a
 * constant, not a learnable parameter or tensor that requires gradients.
 * During backpropagation, the incoming gradient @f$\frac{\partial
 * L}{\partial C}@f$ is passed unchanged to the parent tensor according to the
 * chain rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C}
 * @f]
 *
 * This GradFunction saves the scalar value from the forward pass, though it is
 * not strictly required for the gradient computation. The scalar is stored for
 * potential use in higher-order derivative computations or for consistency
 * with the gradient function interface.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradSubtract for the gradient function for elementwise tensor
 * subtraction.
 */
class GradSubtractScalar : public GradFunction {
 private:
  /// The scalar value from the forward pass, saved for potential use in
  /// higher-order derivative computation
  double savedScalar;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradSubtractScalar gradient function
   *
   * Creates a GradSubtractScalar object that stores the scalar value from the
   * forward pass and a pointer to the parent tensor's GradFunction for
   * gradient propagation during backpropagation.
   *
   * @param savedScalar The scalar value that was subtracted from the tensor
   * during the forward pass
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradSubtractScalar(double savedScalar,
                     std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the tensor-scalar subtraction
   * operation
   *
   * Implements the backward pass for tensor-scalar subtraction by passing the
   * incoming gradient unchanged to the parent tensor. Since the derivative of
   * subtraction with respect to the tensor input is unity and the scalar is
   * treated as a constant, the gradient is propagated without modification.
   * The method calls backward() on the parent tensor's GradFunction to
   * continue the backpropagation process.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the subtraction operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
