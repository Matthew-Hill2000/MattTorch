#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for tensor-scalar multiplication.
 *
 * The GradMultiplyScalar class represents the multiplication of a tensor by a
 * scalar value in the computational graph, computing gradients for operations
 * of the form @f$C = A \cdot s@f$ where @f$A@f$ is a tensor and @f$s@f$ is a
 * scalar value. The gradient computation follows from the derivative:
 * @f[
 *     \frac{\partial C}{\partial A} = s
 * @f]
 *
 * The scalar value does not participate in gradient computation as it is a
 * constant, not a learnable parameter or tensor that requires gradients.
 * During backpropagation, the incoming gradient @f$\frac{\partial
 * L}{\partial C}@f$ is multiplied by the scalar value according to the chain
 * rule:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot
 * \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} \cdot s
 * @f]
 *
 * This GradFunction saves the scalar value from the forward pass as it is
 * required to compute the gradient with respect to the tensor input during the
 * backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradMultiply for the gradient function for elementwise tensor
 * multiplication.
 */
class GradMultiplyScalar : public GradFunction {
 private:
  /// The scalar value from the forward pass, required for computing the
  /// gradient during backpropagation
  double savedScalar;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradMultiplyScalar gradient function
   *
   * Creates a GradMultiplyScalar object that stores the scalar value from the
   * forward pass and a pointer to the parent tensor's GradFunction for
   * gradient propagation during backpropagation.
   *
   * @param savedScalar The scalar value that was multiplied with the tensor
   * during the forward pass
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradMultiplyScalar(double savedScalar,
                     std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the tensor-scalar
   * multiplication operation
   *
   * Implements the backward pass for tensor-scalar multiplication by
   * multiplying the incoming gradient by the saved scalar value. Since the
   * derivative of scalar multiplication with respect to the tensor input is
   * the scalar itself, the gradient is scaled by this value. The method calls
   * backward() on the parent tensor's GradFunction to continue the
   * backpropagation process.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the multiplication operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}
