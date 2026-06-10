#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/mattTorch.h>

#include <memory>

namespace mattTorch::function {

/**
 * @brief Gradient function for the Rectified Linear Unit (ReLU) activation
 * function.
 *
 * The GradReLU class represents the ReLU activation function in the
 * computational graph, computing gradients for the elementwise operation that
 * outputs the maximum of zero and the input. For the operation @f$C =
 * \text{ReLU}(A) = \max(0, A)@f$, the gradient computation follows from the
 * piecewise derivative:
 * @f[
 *     \frac{\partial C}{\partial A} = \begin{cases}
 *       1 & \text{if } A > 0 \\
 *       0 & \text{if } A \leq 0
 *     \end{cases}
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is multiplied elementwise with a mask indicating which elements of the
 * input were positive according to the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A_i} = \begin{cases}
 *       \frac{\partial L}{\partial C_i} & \text{if } A_i > 0 \\
 *       0 & \text{if } A_i \leq 0
 *     \end{cases}
 * @f]
 *
 * This GradFunction saves a backward mask tensor computed during the forward
 * pass, which contains 1 for positive elements and 0 for non-positive
 * elements. The mask is used to efficiently apply the gradient gating during
 * the backward pass without recomputing comparisons.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradTanh for the gradient function for the hyperbolic tangent
 * activation.
 */
class GradReLU : public GradFunction {
 private:
  /// A binary mask tensor created during the forward pass, containing 1 for
  /// elements where the input was positive and 0 for non-positive elements
  Tensor backwardMask;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradReLU gradient function
   *
   * Creates a GradReLU object that stores the backward mask computed during
   * the forward pass and a pointer to the parent tensor's GradFunction for
   * gradient propagation during backpropagation.
   *
   * @param backwardMask A binary mask tensor created during the forward pass,
   * containing 1 for positive input elements and 0 for non-positive elements
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradReLU(Tensor backwardMask,
           std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the ReLU activation
   *
   * Implements the backward pass for the ReLU activation by multiplying the
   * incoming gradient elementwise with the saved backward mask. This
   * effectively passes through the gradient for elements where the input was
   * positive and zeros out the gradient for elements where the input was
   * non-positive, reflecting the piecewise linear nature of ReLU. The method
   * calls backward() on the parent tensor's GradFunction to continue the
   * backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * backward mask to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the ReLU activation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on the backward mask
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
