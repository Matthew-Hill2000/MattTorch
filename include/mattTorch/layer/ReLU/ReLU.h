#pragma once
#include <mattTorch/layer/layer.h>
namespace mattTorch {

/**
 * @brief A Rectified Linear Unit activation layer.
 *
 * The ReLU class implements the ReLU activation function as a network layer,
 * applying the elementwise transformation:
 * @f[
 *     f(x) = \max(0, x)
 * @f]
 * to the input tensor during the forward pass. The input tensor and its shape
 * are stored internally for reference. This layer contains no learnable
 * parameters.
 *
 * @see Layer for the base class from which this class derives.
 * @see TensorView::ReLU() for the underlying tensor operation that performs
 * the elementwise ReLU computation and constructs the computational graph
 * node.
 */
class ReLU : public Layer {
 private:
  /// The input tensor passed to this layer during the most recent forward pass
  TensorView inputTensor;

  /// The shape of the input tensor from the most recent forward pass
  std::vector<int> inputShape;

  /// The output tensor produced by this layer during the most recent forward
  /// pass
  TensorView outputTensor;

 public:
  /// @brief Default constructor
  ReLU();

  /**
   * @brief Performs the forward pass of this ReLU activation layer
   *
   * Applies the ReLU function elementwise to the input tensor by delegating
   * to TensorView::ReLU(). The input tensor and its shape are stored
   * internally before the operation is applied.
   *
   * @param inputTensor The input tensor to apply ReLU activation to
   * @return A TensorView containing the output with ReLU applied to all
   * elements
   */
  TensorView forward(TensorView& inputTensor) override;

  /**
   * @brief Returns an empty vector as this layer has no learnable parameters
   *
   * @return An empty vector
   */
  std::vector<std::shared_ptr<TensorView>> getParameters() override;

  /**
   * @brief Returns zero as this layer has no learnable parameters
   *
   * @return 0
   */
  int getNumParameters() override;
};
}  // namespace mattTorch
