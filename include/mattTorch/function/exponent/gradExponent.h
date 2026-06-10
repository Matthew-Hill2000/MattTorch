#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for raising a tensor to an integer power.
 *
 * The GradExponent class represents the exponentiation operation in the
 * computational graph, computing gradients for raising a tensor to an integer
 * power. For the operation @f$C = A^n@f$ where @f$A@f$ is a tensor and @f$n@f$
 * is an integer exponent, the gradient computation follows from the power rule:
 * @f[
 *     \frac{\partial C}{\partial A} = n \cdot A^{n-1}
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with the derivative according to the chain
 * rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \odot (n
 * \cdot A^{n-1})
 * @f]
 *
 * This GradFunction saves the input tensor and the integer exponent from the
 * forward pass as both are required to compute the gradient during the backward
 * pass using the power rule.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradExponential for the gradient function for the natural exponential
 * function.
 */
class GradExponent : public GradFunction {
 private:
  /// The integer exponent from the forward pass, stored in a vector for
  /// consistency with the gradient function interface
  std::vector<int> savedScalars;

  /// The input tensor from the forward pass, required for computing the
  /// gradient using the power rule
  std::vector<Tensor> savedTensors;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradExponent gradient function
   *
   * Creates a GradExponent object that stores the integer exponent, the input
   * tensor from the forward pass, and a pointer to the parent tensor's
   * GradFunction for gradient propagation during backpropagation.
   *
   * @param savedScalars A vector containing the integer exponent used during
   * the forward pass
   * @param savedTensors A vector containing the input tensor that was raised
   * to a power during the forward pass
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradExponent(std::vector<int> savedScalars, std::vector<Tensor> savedTensors,
               std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the exponentiation operation
   *
   * Implements the backward pass for exponentiation using the power rule. The
   * gradient is computed by raising the input tensor to the power (n-1),
   * multiplying by the exponent n, and then multiplying elementwise with the
   * incoming gradient. The method calls backward() on the parent tensor's
   * GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensor to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the exponentiation operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
