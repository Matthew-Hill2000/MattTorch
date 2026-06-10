#pragma once
#include <memory>
#include <vector>

namespace mattTorch {

class Tensor;

/**
 * @brief Abstract base class for gradient computation nodes in the
 * computational graph.
 *
 * The GradFunction class forms the foundation of mattTorch's automatic
 * differentiation system. Each GradFunction object represents a node in the
 * computational graph corresponding to a mathematical operation performed on
 * one or more tensors during the forward pass. These nodes store the
 * information necessary to compute gradients during the backward pass and
 * maintain pointers to the GradFunction objects of parent tensors in the
 * graph, forming a directed acyclic graph structure that encodes the history
 * of operations.
 *
 * During backpropagation, gradients flow backwards through the computational
 * graph via recursive calls to the backward() method. Each GradFunction
 * receives the gradient of the loss with respect to its output tensor,
 * computes the gradient with respect to each of its input tensors using the
 * chain rule, and propagates these gradients to the parent nodes by calling
 * their backward() methods. This process continues recursively until gradients
 * have been computed for all tensors in the graph that require them.
 *
 * Specific mathematical operations such as addition, multiplication, or
 * activation functions are implemented as derived classes that override the
 * backward() method with the appropriate gradient computation for that
 * operation. Each derived class stores any tensors or values from the forward
 * pass that are needed to compute gradients during the backward pass.
 *
 * @see Tensor for the tensor class that uses GradFunction objects to
 * enable automatic differentiation.
 * @see GradAccumulator for the GradFunction subclass used for leaf tensors.
 */
class GradFunction {
 private:
  /// Stores pointers to the GradFunction objects of the parent tensors in the
  /// computational graph
  std::vector<std::shared_ptr<GradFunction>> nextFunctions;

 public:
  GradFunction() = default;

  /**
   * @brief Computes gradients with respect to parent tensors and propagates
   * them backwards through the computational graph
   *
   * Pure virtual method to be implemented by derived classes with the specific
   * gradient computation for their associated mathematical operation. The
   * method receives the gradient of the loss with respect to this operation's
   * output, computes the gradients with respect to each input tensor using the
   * chain rule, and recursively calls backward() on each parent's GradFunction
   * to continue the backpropagation process.
   *
   * When higherDerivative is false, the implementation should disable gradient
   * tracking on saved tensors by setting requiresGrad to false, preventing the
   * construction of a new computational graph during the backward pass. When
   * higherDerivative is true, operations performed during gradient computation
   * participate in a new computational graph, enabling the computation of
   * higher-order derivatives.
   *
   * @param inputGradient The gradient of the loss with respect to this
   * operation's output tensor
   * @param higherDerivative If true, enables construction of a computational
   * graph during the backward pass to allow computation of higher-order
   * derivatives; if false, disables gradient tracking on operations performed
   * during backpropagation
   */
  virtual void backward(Tensor& inputGradient, bool higherDerivative) = 0;

  /**
   * @brief Virtual destructor for safe polymorphic destruction
   */
  virtual ~GradFunction() = default;

  // Copy Constructor
  GradFunction(GradFunction const& gradfunction) = default;

  // Copy Assignment Operator
  GradFunction& operator=(const GradFunction& gradFunction) = default;

  // Move Constructor
  GradFunction(GradFunction&& gradFunction) = default;

  // Move Assigment Operator
  GradFunction& operator=(GradFunction&& gradFunction) = default;
};

}  // namespace mattTorch
