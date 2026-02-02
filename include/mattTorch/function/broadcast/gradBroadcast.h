#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/mattTorch.h>

#include <memory>

namespace mattTorch::function {

/**
 * @brief Gradient function for broadcasting a tensor along a new dimension.
 *
 * The GradBroadcast class represents the broadcasting operation in the
 * computational graph, computing gradients for expanding a tensor by adding a
 * new dimension and replicating the tensor's values along that dimension. For
 * the operation where tensor @f$A@f$ of shape @f$(d_1, \ldots, d_n)@f$ is
 * broadcast to tensor @f$C@f$ of shape @f$(d_1, \ldots, d_{pos}, N, d_{pos+1},
 * \ldots, d_n)@f$ at position @f$pos@f$, the gradient computation follows
 * from:
 * @f[
 *     \frac{\partial C_{i_1,\ldots,i_{pos},j,i_{pos+1},\ldots,i_n}}{\partial
 * A_{i_1,\ldots,i_n}} = 1
 * @f]
 * for all @f$j@f$.
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial
 * L}{\partial C}@f$ (which has the broadcast dimension) must be summed along
 * the broadcast dimension to match the shape of the input tensor according to
 * the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A_{i_1,\ldots,i_n}} = \sum_{j}
 * \frac{\partial L}{\partial C_{i_1,\ldots,i_{pos},j,i_{pos+1},\ldots,i_n}}
 * @f]
 *
 * The backward pass performs a sum reduction along the broadcast dimension,
 * effectively gathering gradients from all the replicated copies back to the
 * original tensor. This operation is the inverse of the forward broadcasting.
 *
 * This GradFunction saves the input tensor from the forward pass to determine
 * the shape for gradient reduction, and stores the position of the broadcast
 * dimension.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradReductionSum for the gradient function for sum reduction along a
 * dimension.
 */
class GradBroadcast : public GradFunction {
 private:
  /// The input tensor from the forward pass, saved to determine the shape for
  /// gradient reduction
  TensorView savedTensor;

  /// The position (dimension index) at which broadcasting was performed in the
  /// forward pass
  int broadcastPos;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradBroadcast gradient function
   *
   * Creates a GradBroadcast object that stores the input tensor from the
   * forward pass, the position of the broadcast dimension, and a pointer to
   * the parent tensor's GradFunction for gradient propagation during
   * backpropagation.
   *
   * @param savedTensor The input tensor from the forward pass, required for
   * determining the shape for gradient reduction
   * @param broadcastPos The position (dimension index) at which broadcasting
   * was performed in the forward pass
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradBroadcast(TensorView savedTensor, int broadcastPos,
                std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the broadcasting operation
   *
   * Implements the backward pass for broadcasting by summing the incoming
   * gradient along the broadcast dimension to match the shape of the input
   * tensor. This gathers gradients from all replicated copies of the tensor
   * back to the original shape, reflecting that each element of the input
   * contributed to multiple elements of the output. The method calls backward()
   * on the parent tensor's GradFunction to continue the backpropagation
   * process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensor to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * of the broadcasting operation
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
