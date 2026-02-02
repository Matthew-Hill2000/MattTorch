#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for matrix multiplication with one transposed
 * operand.
 *
 * The GradTransposeMatrix class represents matrix multiplication operations
 * where one of the input matrices is transposed during the forward pass,
 * computing gradients for operations of the form @f$C = A^T B@f$ or @f$C = A
 * B^T@f$. The transposeFirst flag determines which operand was transposed.
 *
 * For the operation @f$C = A^T B@f$ (transposeFirst = true), the gradient
 * computation follows from:
 * @f[
 *     \frac{\partial L}{\partial A} = B \left(\frac{\partial L}{\partial
 * C}\right)^T
 * @f]
 * @f[
 *     \frac{\partial L}{\partial B} = A \frac{\partial L}{\partial C}
 * @f]
 *
 * For the operation @f$C = A B^T@f$ (transposeFirst = false), the gradient
 * computation follows from:
 * @f[
 *     \frac{\partial L}{\partial A} = \frac{\partial L}{\partial C} B
 * @f]
 * @f[
 *     \frac{\partial L}{\partial B} = \left(\frac{\partial L}{\partial
 * C}\right)^T A
 * @f]
 *
 * The implementation uses a combination of matrixMultiply and transposeMultiply
 * operations to efficiently compute the required gradient transformations
 * without explicitly materializing transposed matrices where possible.
 *
 * This GradFunction saves copies of both input tensors and a flag indicating
 * which operand was transposed, as all are required to compute the gradients
 * with respect to each input during the backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradMultiplyMatrix for the gradient function for standard matrix
 * multiplication.
 */
class GradTransposeMatrix : public GradFunction {
 private:
  /// The input matrices from the forward pass, required for computing
  /// gradients during backpropagation
  std::vector<TensorView> savedTensors;

  /// Pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

  /// Flag indicating which operand was transposed: true if the first operand
  /// was transposed (A^T * B), false if the second operand was transposed (A *
  /// B^T)
  bool transposeFirst;

 public:
  /**
   * @brief Constructs a GradTransposeMatrix gradient function
   *
   * Creates a GradTransposeMatrix object that stores the input matrices from
   * the forward pass, pointers to the parent GradFunction objects, and a flag
   * indicating which operand was transposed during the forward pass.
   *
   * @param savedTensors A vector containing the two input matrices that
   * participated in the transposed multiplication during the forward pass
   * @param nextFunctions A vector containing shared pointers to the
   * GradFunction objects of the parent tensors for gradient propagation
   * @param transposeFirst Flag indicating which matrix was transposed: true if
   * the first matrix was transposed, false if the second matrix was transposed
   */
  GradTransposeMatrix(std::vector<TensorView> savedTensors,
                      std::vector<std::shared_ptr<GradFunction>> nextFunctions,
                      bool transposeFirst);

  /**
   * @brief Computes and propagates gradients for the transposed matrix
   * multiplication operation
   *
   * Implements the backward pass for transposed matrix multiplication by
   * computing the appropriate matrix products for each parent based on which
   * operand was transposed. The gradient computation differs depending on
   * whether the first or second operand was transposed during the forward
   * pass, with the transposeFirst flag controlling the branching logic. The
   * method uses a combination of matrixMultiply and transposeMultiply
   * operations for efficient computation and calls backward() on each parent's
   * GradFunction to continue the backpropagation process.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the transposed matrix multiplication operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
