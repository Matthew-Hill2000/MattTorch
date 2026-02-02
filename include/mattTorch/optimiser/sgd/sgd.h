#pragma once
#include <mattTorch/network/network.h>
#include <mattTorch/optimiser/optimiser.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <memory>
#include <vector>
namespace mattTorch {

/**
 * @brief Stochastic Gradient Descent optimiser for updating network parameters.
 *
 * The SGD class implements the standard Stochastic Gradient Descent
 * optimisation algorithm, in which each parameter is updated by subtracting
 * its gradient scaled by the learning rate. The parameter update rule is:
 * @f[
 *     \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
 * @f]
 * where @f$\theta_t@f$ is the parameter value at step t, @f$\alpha@f$ is the
 * learning rate, and @f$\nabla L(\theta_t)@f$ is the gradient of the loss
 * with respect to the parameter.
 *
 * The parameter update is performed using AVX2 SIMD intrinsics, processing
 * 4 doubles at a time via fused negative multiply-add (FNMADD) instructions
 * to compute (parameter - learningRate * gradient) in a single operation.
 *
 * @see Optimiser for the base class from which this class derives.
 * @see TensorView for the tensor class representing learnable parameters.
 */
class SGD : public Optimiser {
 private:
  /// Shared pointers to the TensorView objects representing the learnable
  /// parameters of the network to be optimised
  std::vector<std::shared_ptr<TensorView>> parameters;

  /// The learning rate controlling the step size of parameter updates
  double learningRate;

 public:
  /**
   * @brief Constructs an SGD optimiser for the given parameters and learning
   * rate
   *
   * @param parameters A vector of shared pointers to the TensorView objects
   * representing the learnable parameters of the network to be optimised
   * @param learningRate The learning rate controlling the step size of
   * parameter updates
   */
  SGD(std::vector<std::shared_ptr<TensorView>> parameters, double learningRate);

  /**
   * @brief Updates all parameters using Stochastic Gradient Descent
   *
   * Iterates over all parameter tensors and subtracts the gradient scaled by
   * the learning rate from each parameter. The learning rate is broadcast
   * into an AVX2 register and the main loop processes 4 doubles at a time
   * using fused negative multiply-add (FNMADD) intrinsics to compute
   * (parameter - learningRate * gradient) in a single operation.
   */
  void updateParameters();

  /**
   * @brief Resets the gradients of all parameters to zero
   *
   * Iterates over all parameter tensors and sets their gradient data to zero
   * using std::fill. This should be called before each training iteration to
   * prevent gradient accumulation across batches.
   */
  void zeroGrad();
};
}  // namespace mattTorch
