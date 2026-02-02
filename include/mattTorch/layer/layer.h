#pragma once
#include "mattTorch/tensor/tensorView/tensorView.h"

namespace mattTorch {

/**
 * @brief Abstract base class for all neural network layers.
 *
 * The Layer class defines the interface that all layers in a mattTorch neural
 * network must implement. Each layer must provide a forward pass method that
 * takes an input tensor and returns an output tensor, a method to report the
 * number of learnable parameters it contains, and a method to return shared
 * pointers to its learnable parameter tensors. The forward pass of each layer
 * participates in the computational graph, allowing gradients to be propagated
 * back through the layer during backpropagation.
 *
 * @see Network for the class that holds and sequences layers during forward
 * propagation.
 * @see TensorView for the tensor class used to represent inputs, outputs and
 * learnable parameters.
 */
class Layer {
 public:
  /**
   * @brief Performs the forward pass of this layer
   *
   * Pure virtual method to be implemented by derived classes with the specific
   * forward computation for the layer. The returned tensor participates in the
   * computational graph such that gradients can be propagated back through this
   * layer during backpropagation.
   *
   * @param inputTensor The input tensor to pass through the layer
   * @return A TensorView containing the output of the forward pass
   */
  virtual TensorView forward(TensorView& inputTensor) = 0;

  /**
   * @brief Returns the total number of learnable parameters in this layer
   *
   * @return The number of learnable parameters
   */
  virtual int getNumParameters() = 0;

  /**
   * @brief Returns shared pointers to all learnable parameter tensors in this
   * layer
   *
   * @return A vector of shared pointers to the TensorView objects representing
   * the learnable parameters of this layer
   */
  virtual std::vector<std::shared_ptr<TensorView>> getParameters() = 0;

  /**
   * @brief Virtual destructor for safe polymorphic destruction
   */
  virtual ~Layer() {}
};
}  // namespace mattTorch
