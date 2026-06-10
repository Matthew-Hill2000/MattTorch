#pragma once

#include <mattTorch/function/gradFunction.h>

#include "mattTorch/tensor/tensor/tensor.h"

namespace mattTorch::function {

/**
 * @brief Gradient function for the hyperbolic tangent activation function.
 *
 * The GradTanh class represents the hyperbolic tangent activation function in
 * the computational graph, computing gradients for the elementwise tanh
 * operation. For the operation @f$C = \tanh(A)@f$, the gradient computation
 * follows from the derivative:
 * @f[
 *     \frac{\partial C}{\partial A} = 1 - \tanh^2(A) = 1 - C^2
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with @f$1 - C^2@f$ according to the chain
 * rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \odot (1 -
 * C^2)
 * @f]
 *
 * This GradFunction saves the input tensor from the forward pass rather than
 * the output. The tanh function is recomputed during the backward pass to
 * calculate the gradient. This design choice may have been made to maintain
 * consistency across gradient function implementations or to enable specific
 * higher-order derivative computations.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradReLU for the gradient function for the ReLU activation.
 */
class GradTanh : public GradFunction {
 private:
  /// The input tensor from the forward pass, saved for recomputing tanh during
  /// gradient calculation
  Tensor savedTensor;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradTanh gradient function
   *
   * Creates a GradTanh object that stores the input tensor from the forward
   * pass and a pointer to the parent tensor's GradFunction for gradient
   * propagation during backpropagation.
   *
   * @param tanhOutput The input tensor from the forward pass, which will be
   * used to recompute tanh during gradient calculation
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradTanh(Tensor tanhOutput,
           std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the tanh activation
   *
   * Implements the backward pass for the hyperbolic tangent activation by
   * computing the gradient using the formula @f$1 - \tanh^2(x)@f$. The tanh
   * function is recomputed from the saved input tensor, squared, and
   * subtracted from 1 to obtain the local gradient. This is then multiplied
   * elementwise with the incoming gradient. The method calls backward() on the
   * parent tensor's GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensor to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the tanh activation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};

}  // namespace mattTorch::function
