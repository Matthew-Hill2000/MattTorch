#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for elementwise tensor subtraction.
 *
 * The GradSubtract class represents the subtraction operation in the
 * computational graph, computing gradients for the elementwise difference of
 * two tensors. For the operation @f$C = A - B@f$, the gradient computation
 * follows from the derivatives:
 * @f[
 *     \frac{\partial C}{\partial A} = 1, \quad \frac{\partial C}{\partial B}
 * = -1
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is passed to both parent tensors according to the chain rule, with the
 * gradient being negated for the second operand:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C}
 * @f]
 * @f[
 *     \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} \cdot
 * \frac{\partial C}{\partial B} = -\frac{\partial L}{\partial C}
 * @f]
 *
 * This GradFunction saves copies of both input tensors, though they are not
 * strictly required for the gradient computation of subtraction. The saved
 * tensors may be used for higher-order derivative computations or for
 * consistency with the gradient function interface.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradSubtractScalar for the gradient function for tensor-scalar
 * subtraction.
 */
class GradSubtract : public GradFunction {
 private:
  /// The input tensors from the forward pass, saved for potential use in
  /// higher-order derivative computation
  std::vector<TensorView> savedTensors;

  /// Pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradSubtract gradient function
   *
   * Creates a GradSubtract object that stores the input tensors from the
   * forward pass and pointers to the parent GradFunction objects for gradient
   * propagation during backpropagation.
   *
   * @param savedTensors A vector containing the two input tensors that
   * participated in the subtraction during the forward pass
   * @param nextFunctions A vector containing shared pointers to the
   * GradFunction objects of the parent tensors for gradient propagation
   */
  GradSubtract(std::vector<TensorView> savedTensors,
               std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the subtraction operation
   *
   * Implements the backward pass for elementwise subtraction by passing the
   * incoming gradient unchanged to the first parent tensor and negating it for
   * the second parent tensor. Since the derivative of subtraction with respect
   * to the first input is +1 and with respect to the second input is -1, the
   * gradient is propagated accordingly. The method calls backward() on each
   * parent's GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensors to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the subtraction operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
