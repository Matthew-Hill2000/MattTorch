#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for matrix multiplication.
 *
 * The GradMultiplyMatrix class represents the matrix multiplication operation
 * in the computational graph, computing gradients for the product of two
 * matrices. For the operation @f$C = AB@f$ where @f$A@f$ and @f$B@f$ are
 * matrices, the gradient computation follows from the matrix calculus
 * derivatives:
 * @f[
 *     \frac{\partial C}{\partial A} = \frac{\partial L}{\partial C} B^T
 * @f]
 * @f[
 *     \frac{\partial C}{\partial B} = A^T \frac{\partial L}{\partial C}
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * C}@f$ is used to compute gradients with respect to both input matrices
 * according to the chain rule. For the left matrix @f$A@f$, the gradient is
 * computed by multiplying the incoming gradient with the transpose of the
 * right matrix. For the right matrix @f$B@f$, the gradient is computed by
 * multiplying the transpose of the left matrix with the incoming gradient.
 *
 * The implementation uses the transposeMultiply operation for efficient
 * computation of these matrix products, avoiding explicit transposition
 * operations where possible for improved performance.
 *
 * This GradFunction saves copies of both input tensors as they are required to
 * compute the gradients with respect to each input during the backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradMultiply for the gradient function for elementwise multiplication.
 * @see GradTransposeMatrix for the gradient function for transposed matrix
 * multiplication.
 */
class GradMultiplyMatrix : public GradFunction {
 private:
  /// The input matrices from the forward pass, required for computing
  /// gradients during backpropagation
  std::vector<Tensor> savedTensors;

  /// Pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradMultiplyMatrix gradient function
   *
   * Creates a GradMultiplyMatrix object that stores the input matrices from
   * the forward pass and pointers to the parent GradFunction objects for
   * gradient propagation during backpropagation.
   *
   * @param savedTensors A vector containing the two input matrices that were
   * multiplied during the forward pass
   * @param nextFunctions A vector containing shared pointers to the
   * GradFunction objects of the parent tensors for gradient propagation
   */
  GradMultiplyMatrix(std::vector<Tensor> savedTensors,
                     std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the matrix multiplication
   * operation
   *
   * Implements the backward pass for matrix multiplication by computing the
   * appropriate matrix products for each parent. For the left matrix, the
   * gradient is computed as the incoming gradient multiplied by the transpose
   * of the right matrix. For the right matrix, the gradient is computed as the
   * transpose of the left matrix multiplied by the incoming gradient. The
   * transposeMultiply operation is used for efficient computation. The method
   * calls backward() on each parent's GradFunction to continue the
   * backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensors to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the matrix multiplication operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
