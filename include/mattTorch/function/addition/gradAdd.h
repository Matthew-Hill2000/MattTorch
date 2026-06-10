#pragma once

#include <mattTorch/function/gradFunction.h>

#include <memory>

namespace mattTorch::function {

/**
 * @brief Gradient function for elementwise tensor addition.
 *
 * The GradAdd class represents the addition operation in the computational
 * graph, computing gradients for the elementwise sum of two tensors. For the
 * operation @f$C = A + B@f$, the gradient computation follows from the
 * identity:
 * @f[
 *     \frac{\partial C}{\partial A} = 1, \quad \frac{\partial C}{\partial B}
 * = 1
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is passed unchanged to both parent tensors according to the chain
 * rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C}
 * @f]
 * @f[
 *     \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} \cdot
 * \frac{\partial C}{\partial B} = \frac{\partial L}{\partial C}
 * @f]
 *
 * This GradFunction saves copies of both input tensors, though they are not
 * strictly required for the gradient computation of addition. The saved
 * tensors may be used for higher-order derivative computations or for
 * consistency with the gradient function interface.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradAddScalar for the gradient function for tensor-scalar addition.
 */
class GradAdd : public GradFunction {
 private:
  /// Pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradAdd gradient function
   *
   * Creates a GradAdd object that stores the input tensors from the forward
   * pass and pointers to the parent GradFunction objects for gradient
   * propagation during backpropagation.
   *
   * @param nextFunctions A vector containing shared pointers to the
   * GradFunction objects of the parent tensors for gradient propagation
   */
  GradAdd(std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the addition operation
   *
   * Implements the backward pass for elementwise addition by passing the
   * incoming gradient unchanged to both parent tensors. Since the derivative
   * of addition with respect to each input is unity, the gradient is
   * propagated without modification. The method calls backward() on each
   * parent's GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensors to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the addition operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
