#pragma once
#include <mattTorch/criterion/criterion.h>
#include <mattTorch/tensor/tensor/tensor.h>
namespace mattTorch::criterion {

/**
 * @brief Cross-Entropy loss function for classification tasks.
 *
 * The CrossEntropyLoss class computes the Cross-Entropy loss between the
 * predicted values and the target values. The loss is calculated as:
 * @f[
 *     L = -\sum_{i=1}^{N} t_i \log(\hat{y}_i)
 * @f]
 * where @f$t_i@f$ is the target value and @f$\hat{y}_i@f$ is the predicted
 * value. The input is expected to contain probabilities (e.g. the output of
 * a softmax layer) and the target is expected to be a one-hot encoded vector or
 * vector of probabilities. The computation is performed by first taking the
 * elementwise natural logarithm of the input tensor, multiplying elementwise
 * with the target tensor, negating the result, and summing along dimension 1.
 * Each of these operations participates in the computational graph, allowing
 * gradients to be propagated back through the loss function during
 * backpropagation.
 *
 * @see Criterion for the base class from which this class derives.
 */
class CrossEntropyLoss : public Criterion {
 public:
  /**
   * @brief Computes the Cross-Entropy loss between the input predictions
   * and the target values
   *
   * @param input The predicted probability values produced by the network,
   * expected to be the output of a softmax layer
   * @param target The ground truth target values as one-hot encoded vectors
   * @return A Tensor containing the computed Cross-Entropy loss value
   */
  Tensor calculateLoss(Tensor& input, Tensor& target) override;
};
}  // namespace mattTorch::criterion
