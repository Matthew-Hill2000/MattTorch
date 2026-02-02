#pragma once

#include <mattTorch/function/gradFunction.h>


namespace mattTorch::function {

/**
 * @brief Gradient function for elementwise tensor division.
 *
 * The GradDivide class represents the elementwise division operation in the
 * computational graph, computing gradients for the quotient of two tensors.
 * For the operation @f$C = A / B@f$ or equivalently @f$C = A \oslash B@f$
 * where @f$\oslash@f$ denotes elementwise division, the gradient computation
 * follows from the derivatives:
 * @f[
 *     \frac{\partial C}{\partial A} = \frac{1}{B}, \quad \frac{\partial
 * C}{\partial B} = -\frac{A}{B^2}
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with the appropriate derivative according to
 * the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \oslash B
 * @f]
 * @f[
 *     \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial B} = -\frac{\partial L}{\partial C} \odot
 * \frac{A}{B^2}
 * @f]
 *
 * This GradFunction saves copies of both input tensors as they are required to
 * compute the gradients with respect to each input during the backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradDivideScalar for the gradient function for tensor-scalar division.
 */
class GradDivide : public GradFunction {
private:
  /// The input tensors from the forward pass, required for computing gradients
  /// during backpropagation
  std::vector<TensorView> savedTensors;

  /// Pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

public:
  /**
   * @brief Constructs a GradDivide gradient function
   *
   * Creates a GradDivide object that stores the input tensors from the forward
   * pass and pointers to the parent GradFunction objects for gradient
   * propagation during backpropagation.
   *
   * @param savedTensors A vector containing the two input tensors that
   * participated in the division during the forward pass
   * @param nextFunction A vector containing shared pointers to the
   * GradFunction objects of the parent tensors for gradient propagation
   */
  GradDivide(std::vector<TensorView> savedTensors,
             std::vector<std::shared_ptr<GradFunction>> nextFunction);

  /**
   * @brief Computes and propagates gradients for the elementwise division
   * operation
   *
   * Implements the backward pass for elementwise division by computing the
   * appropriate gradients for each parent. For the numerator, the gradient is
   * computed by dividing the incoming gradient by the denominator. For the
   * denominator, the gradient is computed using the quotient rule, involving
   * both the numerator and denominator. The method calls backward() on each
   * parent's GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensors to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the division operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(TensorView &inputGradient, bool higherDerivative) override;
};
}
