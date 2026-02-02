#pragma once
#include <mattTorch/criterion/criterion.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch::criterion {

/**
 * @brief Mean Squared Error loss function for regression tasks.
 *
 * The MSELoss class computes the Mean Squared Error between the predicted
 * values and the target values. The loss is calculated as:
 * @f[
 *     L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2
 * @f]
 * where @f$y_i@f$ is the target value, @f$\hat{y}_i@f$ is the predicted
 * value, and @f$N@f$ is the total number of elements. The computation is
 * performed by first computing the elementwise difference between the input
 * and target tensors, squaring the difference elementwise, summing along
 * dimension 1, and dividing by the total number of elements. Each of these
 * operations participates in the computational graph, allowing gradients to
 * be propagated back through the loss function during backpropagation.
 *
 * @see Criterion for the base class from which this class derives.
 */
class MSELoss : public Criterion {
 public:
  /**
   * @brief Computes the Mean Squared Error loss between the input predictions
   * and the target values
   *
   * @param input The predicted values produced by the network
   * @param target The ground truth target values
   * @return A TensorView containing the computed MSE loss value
   */
  TensorView calculateLoss(TensorView& input, TensorView& target);
};
}  // namespace mattTorch::criterion
