#pragma once

#include <mattTorch/function/gradFunction.h>


namespace mattTorch::function {

/**
 * @brief Gradient function for tensor-scalar division.
 *
 * The GradDivideScalar class represents division operations involving a tensor
 * and a scalar value in the computational graph, computing gradients for
 * operations of the form @f$C = A / s@f$ or @f$C = s / A@f$ where @f$A@f$ is
 * a tensor and @f$s@f$ is a scalar value. The numerator flag determines which
 * form was used during the forward pass.
 *
 * For the operation @f$C = A / s@f$ (numerator = true), the gradient
 * computation follows from:
 * @f[
 *     \frac{\partial C}{\partial A} = \frac{1}{s}
 * @f]
 * and during backpropagation:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \cdot
 * \frac{1}{s}
 * @f]
 *
 * For the operation @f$C = s / A@f$ (numerator = false), the gradient
 * computation follows from:
 * @f[
 *     \frac{\partial C}{\partial A} = -\frac{s}{A^2}
 * @f]
 * and during backpropagation:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} \odot
 * \left(-\frac{s}{A^2}\right)
 * @f]
 *
 * This GradFunction saves the scalar value, a shared pointer to the tensor
 * from the forward pass, and a flag indicating whether the tensor was the
 * numerator or denominator, as all are required to compute the gradient during
 * the backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradDivide for the gradient function for elementwise tensor division.
 */
class GradDivideScalar : public GradFunction {
 private:
  /// The scalar value from the forward pass, required for computing the
  /// gradient during backpropagation
  double savedScalar;

  /// A shared pointer to the tensor from the forward pass, required for
  /// computing the gradient when the tensor was the denominator
  std::shared_ptr<TensorView> savedTensor;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

  /// Flag indicating whether the tensor was the numerator (true for A/s) or
  /// denominator (false for s/A) in the forward pass
  bool numerator;

 public:
  /**
   * @brief Constructs a GradDivideScalar gradient function
   *
   * Creates a GradDivideScalar object that stores the scalar value, a shared
   * pointer to the tensor from the forward pass, a pointer to the parent
   * tensor's GradFunction, and a flag indicating whether the tensor was the
   * numerator or denominator.
   *
   * @param savedScalar The scalar value that participated in the division
   * during the forward pass
   * @param savedTensor A shared pointer to the tensor from the forward pass,
   * required for computing gradients when the tensor was the denominator
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   * @param numerator Flag indicating the tensor's role: true if the tensor was
   * the numerator (A/s), false if the tensor was the denominator (s/A),
   * defaults to true
   */
  GradDivideScalar(double savedScalar, std::shared_ptr<TensorView> savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions,
                   bool numerator = true);

  /**
   * @brief Computes and propagates gradients for the tensor-scalar division
   * operation
   *
   * Implements the backward pass for tensor-scalar division with behaviour
   * determined by the numerator flag. When the tensor was the numerator, the
   * incoming gradient is divided by the scalar value. When the tensor was the
   * denominator, the gradient is computed using the quotient rule involving
   * the saved tensor and scalar. The method calls backward() on the parent
   * tensor's GradFunction to continue the backpropagation process.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the division operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
