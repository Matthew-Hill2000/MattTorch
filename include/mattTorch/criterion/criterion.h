#pragma once
#include "mattTorch/tensor/tensor/tensor.h"
namespace mattTorch {

/**
 * @brief Abstract base class for all loss functions.
 *
 * The Criterion class defines the interface for computing loss values during
 * neural network training. Specific loss functions such as cross-entropy or
 * mean squared error are intended to derive from this class and implement the
 * calculateLoss method with the appropriate loss computation. The resulting
 * loss tensor participates in the computational graph, allowing gradients to
 * be propagated back through the loss function during backpropagation.
 *
 * @see Tensor for the tensor class used to represent inputs, targets and
 * loss values.
 */
class Criterion {
 public:
  /**
   * @brief Computes the loss between the input predictions and the target
   * values
   *
   * Pure virtual method to be implemented by derived classes with the
   * specific loss computation. The returned tensor participates in the
   * computational graph such that calling backward() on it will propagate
   * gradients back through the network.
   *
   * @param input The predicted values produced by the network
   * @param target The ground truth target values
   * @return A Tensor containing the computed loss value
   */
  virtual Tensor calculateLoss(Tensor& input, Tensor& target) = 0;

  /**
   * @brief Virtual destructor for safe polymorphic destruction
   */
  virtual ~Criterion() = default;
};
}  // namespace mattTorch
