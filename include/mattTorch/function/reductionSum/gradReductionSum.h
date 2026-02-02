#pragma once

#include <mattTorch/function/gradFunction.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for sum reduction along a specified dimension.
 *
 * The GradReductionSum class represents the sum reduction operation in the
 * computational graph, computing gradients for summing a tensor along a
 * specific dimension. For the operation @f$C = \sum_{i} A_i@f$ where the sum
 * is taken along dimension @f$d@f$ of tensor @f$A@f$, the gradient
 * computation follows from:
 * @f[
 *     \frac{\partial C_j}{\partial A_{j,i}} = 1 \quad \text{for all } i
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial
 * L}{\partial C}@f$ (which has one fewer dimension than the input) must be
 * broadcast along the reduced dimension to match the shape of the input
 * tensor according to the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A_{j,i}} = \frac{\partial L}{\partial C_j}
 * \quad \text{for all } i
 * @f]
 *
 * The backward pass broadcasts the incoming gradient along the reduced
 * dimension to restore the original tensor shape, effectively distributing the
 * gradient to all elements that contributed to each sum. This operation is the
 * inverse of the forward sum reduction.
 *
 * This GradFunction saves the input tensor from the forward pass to determine
 * the shape for broadcasting the gradient, and stores the dimension along
 * which the reduction was performed.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradMean for the gradient function for mean reduction.
 * @see GradBroadcast for the gradient function for broadcasting operations.
 */
class GradReductionSum : public GradFunction {
 private:
  /// The input tensor from the forward pass, saved to determine the shape for
  /// gradient broadcasting
  std::vector<TensorView> savedTensors;

  /// The dimension along which the sum reduction was performed in the forward
  /// pass
  int reduceDim;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradReductionSum gradient function
   *
   * Creates a GradReductionSum object that stores the input tensor from the
   * forward pass, the dimension along which reduction was performed, and a
   * pointer to the parent tensor's GradFunction for gradient propagation
   * during backpropagation.
   *
   * @param savedTensors A vector containing the input tensor from the forward
   * pass, required for determining the shape for gradient broadcasting
   * @param reduceDim The dimension along which the sum reduction was performed
   * in the forward pass
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradReductionSum(std::vector<TensorView> savedTensors, int reduceDim,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the sum reduction operation
   *
   * Implements the backward pass for sum reduction by broadcasting the
   * incoming gradient along the reduced dimension to match the shape of the
   * input tensor. This distributes the gradient to all elements that
   * contributed to each sum, reflecting that each element contributes equally
   * (with a derivative of 1) to the sum. The method calls backward() on the
   * parent tensor's GradFunction to continue the backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensors to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the sum reduction operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
