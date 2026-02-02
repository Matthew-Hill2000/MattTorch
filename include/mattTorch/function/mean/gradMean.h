#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::function {

/**
 * @brief Gradient function for computing the mean of all elements in a tensor.
 *
 * The GradMean class represents the mean reduction operation in the
 * computational graph, computing gradients for averaging all elements of a
 * tensor into a single scalar value. For the operation @f$c = \text{mean}(A) =
 * \frac{1}{N}\sum_{i} A_i@f$ where @f$A@f$ is a tensor and @f$N@f$ is the
 * total number of elements, the gradient computation follows from:
 * @f[
 *     \frac{\partial c}{\partial A_i} = \frac{1}{N} \quad \text{for all } i
 * @f]
 *
 * During backpropagation, the incoming gradient @f$\frac{\partial L}{\partial
 * c}@f$ (a scalar) must be distributed equally to all elements of the input
 * tensor according to the chain rule:
 * @f[
 *     \frac{\partial L}{\partial A_i} = \frac{\partial L}{\partial c} \cdot
 * \frac{\partial c}{\partial A_i} = \frac{1}{N} \cdot \frac{\partial
 * L}{\partial c}
 * @f]
 *
 * The backward pass broadcasts the incoming gradient (divided by the number of
 * elements) to match the shape of the input tensor, effectively distributing
 * the gradient equally across all input elements.
 *
 * This GradFunction saves the input tensor from the forward pass to determine
 * the number of elements and the shape for broadcasting the gradient during
 * the backward pass.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see GradReductionSum for the gradient function for sum reduction along a
 * dimension.
 */
class GradMean : public GradFunction {
 private:
  /// The input tensor from the forward pass, saved to determine the number of
  /// elements and shape for gradient broadcasting
  TensorView savedTensor;

  /// Pointer to the GradFunction object of the parent tensor in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  /**
   * @brief Constructs a GradMean gradient function
   *
   * Creates a GradMean object that stores the input tensor from the forward
   * pass and a pointer to the parent tensor's GradFunction for gradient
   * propagation during backpropagation.
   *
   * @param savedTensor The input tensor from the forward pass, required for
   * determining the number of elements and shape for gradient broadcasting
   * @param nextFunctions A vector containing a shared pointer to the
   * GradFunction object of the parent tensor for gradient propagation
   */
  GradMean(TensorView savedTensor,
                   std::vector<std::shared_ptr<GradFunction>> nextFunctions);

  /**
   * @brief Computes and propagates gradients for the mean operation
   *
   * Implements the backward pass for the mean reduction by dividing the
   * incoming scalar gradient by the total number of elements in the input
   * tensor and broadcasting it to match the input tensor's shape. This
   * distributes the gradient equally to all elements of the input tensor,
   * reflecting that each element contributes equally to the mean. The method
   * calls backward() on the parent tensor's GradFunction to continue the
   * backpropagation process.
   *
   * When higherDerivative is false, gradient tracking is disabled on the
   * saved tensor to prevent construction of a new computational graph during
   * backpropagation.
   *
   * @param inputGradient The gradient of the loss with respect to the output
   * scalar mean value
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass for higher-order derivatives; if false,
   * disables gradient tracking on saved tensors
   */
  void backward(TensorView& inputGradient, bool higherDerivative) override;
};
}  // namespace mattTorch::function
