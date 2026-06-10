#pragma once
#include <mattTorch/layer/layer.h>
namespace mattTorch {

/**
 * @brief A Softmax activation layer.
 *
 * The Softmax class implements the softmax activation function as a network
 * layer, computing normalised probabilities across dimension 1 of the input
 * tensor. The softmax function is defined as:
 * @f[
 *     \text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
 * @f]
 * The forward pass first applies the exponential function elementwise to the
 * input tensor, then computes the sum of the exponentials along dimension 1,
 * broadcasts the summed values back to the original shape, and divides the
 * exponentials by the broadcast sums to produce the normalised probabilities.
 * Each of these operations participates in the computational graph, allowing
 * gradients to be propagated back through this layer during backpropagation.
 * This layer contains no learnable parameters.
 *
 * @see Layer for the base class from which this class derives.
 */
class Softmax : public Layer {
 private:
  /// The input tensor passed to this layer during the most recent forward pass
  Tensor inputTensor;

  /// The shape of the input tensor from the most recent forward pass
  Dims inputShape;

  /// The output tensor produced by this layer during the most recent forward
  /// pass
  Tensor outputTensor;

 public:
  /// @brief Default constructor
  Softmax() = default;

  /**
   * @brief Performs the forward pass of this Softmax activation layer
   *
   * Computes the softmax probabilities by first applying the exponential
   * function elementwise to the input tensor, summing the resulting
   * exponentials along dimension 1, broadcasting the sums back to the
   * original shape, and dividing the exponentials by the broadcast sums
   * in place. The input tensor and its shape are stored internally before
   * the operation is applied.
   *
   * @param inputTensor The input tensor to apply Softmax activation to
   * @return A Tensor containing the normalised probability output
   */
  Tensor forward(Tensor& inputTensor) override;

  /**
   * @brief Returns an empty vector as this layer has no learnable parameters
   *
   * @return An empty vector
   */
  std::vector<std::shared_ptr<Tensor>> getParameters() override;

  /**
   * @brief Returns zero as this layer has no learnable parameters
   *
   * @return 0
   */
  int getNumParameters() override;
};
}  // namespace mattTorch
