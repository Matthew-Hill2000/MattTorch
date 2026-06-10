#pragma once
#include <mattTorch/layer/layer.h>
namespace mattTorch {

/**
 * @brief A fully connected (dense) neural network layer.
 *
 * The FullyConnectedLayer class implements a standard dense layer that
 * computes the affine transformation:
 * @f[
 *     Y = XW + b
 * @f]
 * where @f$X@f$ is the input tensor, @f$W@f$ is the weight matrix, and
 * @f$b@f$ is the bias vector. The weight matrix is of shape
 * (inputs x outputs) and the bias vector is of shape (outputs). During the
 * forward pass, the bias vector is broadcast along a new first dimension to
 * match the batch size of the input tensor before being added elementwise to
 * the weighted output.
 *
 * The weight and bias parameters are initialised at construction time using
 * either Xavier or He initialisation, both of which sample from a normal
 * distribution with a variance scaled according to the number of input and
 * output features. Xavier initialisation uses a standard deviation of
 * @f$\sqrt{2 / (n_{in} + n_{out})}@f$ and He initialisation uses a standard
 * deviation of @f$\sqrt{2 / n_{in}}@f$.
 *
 * @see Layer for the base class from which this class derives.
 * @see Tensor for the tensor class used to represent the weights, biases,
 * inputs and outputs.
 */
class FullyConnectedLayer : public Layer {
 private:
  /// The weight matrix of shape (inputs x outputs)
  Tensor weight;

  /// The bias vector of shape (outputs)
  Tensor bias;

 public:
  /**
   * @brief Constructs a fully connected layer with the specified input and
   * output sizes
   *
   * Allocates the weight matrix of shape (inputs x outputs) and the bias
   * vector of shape (outputs), and initialises both using the specified
   * initialisation strategy by sampling from a normal distribution with
   * appropriately scaled variance.
   *
   * @param inputs The number of input features to the layer
   * @param outputs The number of output features from the layer
   * @param initialisation The weight initialisation strategy, either "xavier"
   * or "he", defaults to "xavier"
   *
   * @throws std::invalid_argument If the initialisation string is not "xavier"
   * or "he"
   */
  FullyConnectedLayer(int inputs, int outputs,
                      std::string initialisation = "xavier");

  /**
   * @brief Performs the forward pass of this fully connected layer
   *
   * Computes the affine transformation Y = XW + b by first performing matrix
   * multiplication of the input tensor with the weight matrix, then
   * broadcasting the bias vector along a new first dimension to match the
   * batch size of the input, and finally adding the broadcast bias to the
   * weighted output elementwise. Each of these operations participates in the
   * computational graph, allowing gradients to be propagated back through this
   * layer during backpropagation.
   *
   * @param inputTensor The input tensor of shape (batchSize x inputs)
   * @return A Tensor containing the output of shape (batchSize x outputs)
   */
  Tensor forward(Tensor& inputTensor) override;

  /**
   * @brief Returns the total number of learnable parameters in this layer
   *
   * The total number of parameters is the sum of the number of elements in
   * the weight matrix and the number of elements in the bias vector.
   *
   * @return The total number of learnable parameters
   */
  int getNumParameters() override;

  /**
   * @brief Returns shared pointers to the weight and bias parameter tensors
   *
   * @return A vector containing shared pointers to the weight Tensor and
   * the bias Tensor
   */
  std::vector<std::shared_ptr<Tensor>> getParameters() override;
};
}  // namespace mattTorch
