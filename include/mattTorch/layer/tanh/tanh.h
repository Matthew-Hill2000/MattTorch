#pragma once
#include <mattTorch/layer/layer.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch {

/**
 * @brief A hyperbolic tangent activation layer.
 *
 * The Tanh class implements the tanh activation function as a network layer,
 * applying the elementwise transformation:
 * @f[
 *     f(x) = \tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}
 * @f]
 * to the input tensor during the forward pass by delegating to
 * TensorView::tanh(). This layer contains no learnable parameters.
 *
 * @see Layer for the base class from which this class derives.
 * @see TensorView::tanh() for the underlying tensor operation that performs
 * the elementwise tanh computation and constructs the computational graph
 * node.
 */
class Tanh : public Layer {
 public:
  /// @brief Default constructor
  Tanh() = default;

  /**
   * @brief Performs the forward pass of this Tanh activation layer
   *
   * Applies the tanh function elementwise to the input tensor by delegating
   * to TensorView::tanh().
   *
   * @param input The input tensor to apply Tanh activation to
   * @return A TensorView containing the output with tanh applied to all
   * elements
   */
  TensorView forward(TensorView& input) override;

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
