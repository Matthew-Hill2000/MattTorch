#pragma once
#include <mattTorch/network/network.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <memory>
#include <vector>
namespace mattTorch {

/**
 * @brief Base class for all parameter optimisation algorithms.
 *
 * The Optimiser class provides the interface and common functionality for
 * optimising the parameters of a neural network during training. It holds
 * shared pointers to the Tensor objects representing the learnable
 * parameters of a network, along with a learning rate that controls the
 * step size of parameter updates. Specific optimisation algorithms such as
 * SGD or Adam are intended to derive from this class and override the
 * parameter update logic as needed.
 *
 * @see Network for the class whose parameters are optimised by this class.
 * @see Tensor for the tensor class representing learnable parameters.
 */
class Optimiser {
 private:
  /// Shared pointers to the Tensor objects representing the learnable
  /// parameters of the network to be optimised
  std::vector<std::shared_ptr<Tensor>> parameters;

  /// The learning rate controlling the step size of parameter updates
  double learningRate;

 public:
  /**
   * @brief Updates all parameters using their computed gradients
   *
   * Iterates over all parameter tensors and applies the optimisation step,
   * adjusting each parameter in the direction indicated by its gradient
   * scaled by the learning rate.
   */
  void updateParameters();

  /**
   * @brief Resets the gradients of all parameters to zero
   *
   * Iterates over all parameter tensors and sets their gradients to zero.
   * This should be called before each training iteration to prevent gradient
   * accumulation across batches.
   */
  void zeroGrad();

  /**
   * @brief Virtual destructor for safe polymorphic destruction
   */
  virtual ~Optimiser() = default;
};
}  // namespace mattTorch
