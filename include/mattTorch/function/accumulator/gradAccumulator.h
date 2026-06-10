#pragma once

#include <mattTorch/function/gradFunction.h>
#include <mattTorch/tensor/tensor/tensorTypes.h>

#include "mattTorch/tensor/tensor/tensor.h"

namespace mattTorch::function {

/**
 * @brief Gradient function for leaf tensors that accumulates gradients during
 * backpropagation.
 *
 * The GradAccumulator class is a specialised GradFunction used exclusively for
 * leaf tensors in the computational graph. Unlike other GradFunction objects
 * that compute gradients with respect to their inputs and pass them to parent
 * nodes, the GradAccumulator terminates the backpropagation chain by
 * accumulating the incoming gradient into the leaf tensor's gradient storage.
 * This allows leaf tensors such as learnable parameters in a neural network to
 * collect the total gradient of the loss with respect to themselves.
 *
 * When backward() is called for the first time, the incoming gradient is
 * copied directly into the leaf tensor's gradient. On subsequent calls during
 * the same backward pass or across multiple backward passes, the incoming
 * gradient is accumulated (summed) with the existing gradient. This
 * accumulation behaviour is essential for computing gradients when a leaf
 * tensor is used multiple times in the computational graph, as the total
 * gradient is the sum of all partial gradients according to the multivariable
 * chain rule.
 *
 * When computing higher-order derivatives with higherDerivative set to true,
 * the GradAccumulator enables the accumulated gradient itself to participate
 * in a new computational graph by setting the gradient's GradFunction to match
 * the incoming gradient's GradFunction and enabling gradient tracking on it.
 * This allows derivatives of derivatives to be computed through subsequent
 * backward passes.
 *
 * @see GradFunction for the base class from which this class derives.
 * @see Tensor for the tensor class that uses GradAccumulator for leaf
 * tensors.
 */
class GradAccumulator : public GradFunction {
 private:
  /// A shared pointer to the gradient Tensor belonging to the leaf tensor
  /// that owns this GradAccumulator
  std::shared_ptr<Tensor> gradient;

  /// The dimensions of the leaf tensor
  Dims dims;

 public:
  /**
   * @brief Constructs a GradAccumulator for a leaf tensor
   *
   * Creates a GradAccumulator that will accumulate gradients into the
   * specified gradient tensor. The dimensions of the leaf tensor are stored
   * for potential use in gradient shape validation or manipulation.
   *
   * @param gradient A shared pointer to the Tensor that will store the
   * accumulated gradient for the leaf tensor
   * @param dims The dimensions of the leaf tensor
   */
  GradAccumulator(std::shared_ptr<Tensor> gradient, Dims dims);

  /**
   * @brief Accumulates the incoming gradient into the leaf tensor's gradient
   * storage
   *
   * Implements the gradient accumulation logic for leaf tensors. If this is
   * the first gradient being accumulated (the leaf tensor does not yet have a
   * gradient), the incoming gradient is copied directly into the gradient
   * storage. If the leaf tensor already has a gradient, the incoming gradient
   * is added elementwise to the existing gradient.
   *
   * When higherDerivative is true, the accumulated gradient tensor is
   * configured to participate in a new computational graph by setting its
   * GradFunction to match the incoming gradient's GradFunction and enabling
   * gradient tracking on it. This enables computation of higher-order
   * derivatives. When higherDerivative is false, the addition operation used
   * for accumulation does not construct a computational graph.
   *
   * @param inputGradient The gradient to accumulate, representing the gradient
   * of the loss with respect to this leaf tensor
   * @param higherDerivative If true, enables the accumulated gradient to
   * participate in a new computational graph for higher-order derivative
   * computation; if false, performs accumulation without gradient tracking
   */
  void backward(Tensor& inputGradient, bool higherDerivative) override;

  /**
   * @brief Updates the shared pointer to the gradient storage
   *
   * Allows the gradient tensor pointer to be updated, which may be necessary
   * when gradients are zeroed or reset between training iterations.
   *
   * @param newGrad A shared pointer to the new gradient Tensor
   */
  void setGradient(std::shared_ptr<Tensor> newGrad);
};
}  // namespace mattTorch::function
