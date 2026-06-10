#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for elementwise tensor multiplication.
 *
 * The GradMultiply class represents the elementwise multiplication operation
 * in the computational graph, computing gradients for the Hadamard product of
 * two tensors. For the operation @f$C = A \odot B@f$ where @f$\odot@f$
 * denotes elementwise multiplication, the gradient computation follows from
 * the derivatives:
 * @f[
 *     \frac{\partial C}{\partial A} = B, \quad \frac{\partial C}{\partial B}
 * = A
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with the appropriate saved tensor according
 * to the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \odot B
 * @f]
 * @f[
 *     \frac{\partial L}{\partial B} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial B} = \frac{\partial L}{\partial C} \odot A
 * @f]
 *
 * This GradFunction saves copies of both input tensors as they are required to
 * compute the gradients with respect to each input during the backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradMultiplyScalar for the gradient function for tensor-scalar
 * multiplication.
 * @see GradMultiplyMatrix for the gradient function for matrix multiplication.
 */
class GradMultiply : public GradFunction {
 private:
  /// The input tensors from the forward pass, required for computing gradients
  /// during backpropagation
  std::vector<Tensor> savedTensors;

  /// Pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradMultiply gradient function
   *
   * Creates a GradMultiply object that stores the input tensors from the
   * forward pass and pointers to the parent GradFunction objects for gradient
   * propagation during backpropagation.
   *
   * @param savedTensors A vector containing the two input tensors that were
   * multiplied elementwise during the forward pass
   * @param nextFunctions A vector containing shared pointers to the
   * GradFunction objects of the parent tensors for gradient propagation
   */
  GradMultiply(std::vector<Tensor> savedTensors,
               std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the elementwise
   * multiplication operation
   *
   * Implements the backward pass for elementwise multiplication by multiplying
   * the incoming gradient with the appropriate saved tensor for each parent.
   * For the first parent, the gradient is multiplied elementwise with the
   * second saved tensor; for the second parent, the gradient is multiplied
   * elementwise with the first saved tensor. The method calls backward() on
   * each parent's GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensors to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the multiplication operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
