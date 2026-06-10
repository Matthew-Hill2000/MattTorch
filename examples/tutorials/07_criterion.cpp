#include <mattTorch/mattTorch.h>

#include <iostream>
#include <print>

#include "mattTorch/criterion/crossEntropyLoss/crossEntropyLoss.h"
#include "mattTorch/criterion/mseLoss/mseLoss.h"
#include "mattTorch/network/network.h"
#include "mattTorch/optimiser/sgd/sgd.h"
#include "mattTorch/tensor/tensor/tensor.h"

int main(int argc, char* argv[]) {
  // The previous six tutorials have established all of the machinery needed
  // to train a neural network. A Tensor class provides aligned contiguous
  // storage and the indexing arithmetic that lets it represent
  // multidimensional arrays of any shape. A library of SIMD-vectorised
  // mathematical operations acts on those Tensors and simultaneously builds
  // a directed acyclic computational graph through which gradients can be
  // propagated in a single backward pass. A Layer abstraction packages
  // sequences of operations with their learnable parameters into a unit of
  // composition, and a Network chains those layers into an end-to-end
  // differentiable mapping. An Optimiser consumes the gradients accumulated
  // into the parameter Tensors by the backward pass and updates them in
  // place, ready for the next forward pass.
  //
  // The one remaining building block is the loss function. In tutorial six
  // the loss was assembled from raw Tensor operations directly inside the
  // training loop: a subtraction to form the residuals, an elementwise
  // multiplication to square them, and a mean to collapse them to a scalar.
  // This works correctly, since every one of those operations participates in
  // the autograd system and appends its own GradFunction nodes to the
  // computational graph in exactly the way described in tutorial three. But
  // it places the full responsibility for choosing and correctly implementing
  // the loss computation on whoever writes the training loop, and repeats the
  // same code in every loop that uses the same loss function. The Criterion
  // class exists to encapsulate that responsibility behind a clean, reusable
  // interface that integrates naturally with the rest of the library.
  //
  // The Criterion base class is abstract in exactly the same sense that Layer
  // is. It declares a single pure virtual method, calculateLoss, which takes
  // references to an input Tensor of network predictions and a target Tensor
  // of ground truth values, and returns a Tensor containing the computed loss.
  // Specific loss functions derive from Criterion and provide their own
  // implementation of calculateLoss. The virtual destructor is declared for
  // the same reason as in Layer: a derived Criterion object deleted through a
  // base-class Criterion* must run the correct destructor, which matters
  // whenever a concrete criterion is managed through a pointer to the base
  // class.
  //
  // Two concrete derivations are provided. MSELoss implements mean squared
  // error and is the standard choice for regression tasks. CrossEntropyLoss
  // implements cross-entropy and is the standard choice for multi-class
  // classification tasks. Both are defined in the mattTorch::criterion
  // namespace.

  std::println("-----------------------------------------------------------");

  // MSELoss computes the mean squared error between the network's predictions
  // and the ground truth target values. For a pair of input and target Tensors
  // with a combined total of N elements, the loss is defined as
  //
  //     L = (1 / N) * sum_i (input_i - target_i)^2
  //
  // where the sum runs over every element of the two Tensors. The
  // implementation follows four steps, each of which is a standard Tensor
  // operation. First, the elementwise subtraction operator computes the
  // residual Tensor of the same shape as the input. Second, the residuals are
  // squared elementwise by multiplying the residual Tensor with itself using
  // the elementwise multiplication operator. Third, reductionSum(1) is called
  // on the squared residuals to sum along dimension one, collapsing each row
  // of the (batchSize x features) Tensor into a single summed value per
  // sample and producing a result of shape (batchSize). Fourth, the result is
  // divided by the total number of elements in the input, obtained via
  // getNValues, to apply the 1/N normalisation.
  //
  // Every one of these four steps uses the standard Tensor arithmetic
  // operators described in tutorials two and three. Each operator allocates
  // its own GradFunction on the result and records its operands in
  // savedTensors in exactly the way described for any other Tensor operation.
  // The loss Tensor returned by calculateLoss therefore carries a full
  // computational graph reaching back through the squared residuals and the
  // raw residuals, and through whatever network output was passed as the
  // input argument. Calling backward on it propagates gradients from the
  // loss all the way back through the network layers in the usual way, without
  // any special treatment. The Criterion class does not need to know anything
  // about the graph that produced its input: it simply appends its own nodes
  // to whatever graph already exists on the input Tensor.
  //
  // The target Tensor is expected to have its requiresGrad attribute set to
  // false before being passed to calculateLoss. As described in tutorial
  // three, any operation that sees at least one operand with requiresGrad set
  // to true will allocate a GradFunction node. Marking the target as not
  // requiring grad ensures that the branches of the graph leading into the
  // target are terminated cleanly with null nextFunctions entries rather than
  // reaching into any GradAccumulator that the target might otherwise carry.
  // The result is a graph in which only the prediction side is differentiable,
  // which is exactly what is needed: gradients should flow back into the
  // network parameters, not into the ground truth labels.

  mattTorch::Tensor input({2, 4});
  mattTorch::Tensor target({2, 4});
  target.setRequiresGrad(false);

  for (int i{0}; i < input.getDimensions()[0]; i++) {
    for (int j{0}; j < input.getDimensions()[1]; j++) {
      input[{i, j}] = 0.1 * (i + j + 1);
      target[{i, j}] = 0.1 * (i + j) + 0.05;
    }
  }

  std::cout << "input:\n" << input << "\n";
  std::cout << "target:\n" << target << "\n";

  mattTorch::criterion::MSELoss mse;
  auto mseLoss = mse.calculateLoss(input, target);

  // The loss Tensor has shape (2) after the reductionSum(1) step, one entry
  // per sample in the batch. Each entry is the sum of squared residuals for
  // that sample divided by the total element count N = 2 * 4 = 8. Calling
  // backward with no explicit seed constructs the all-ones Tensor of shape
  // (2) as the initial inputGradient, as described in tutorial three. This
  // is equivalent to differentiating the sum of the two per-sample losses
  // with respect to the input, which equals the gradient of the total mean
  // squared error over every element of the batch. The gradient accumulated
  // into each input element by the backward pass is
  //
  //     dL/d(input[i, j]) = 2 * (input[i, j] - target[i, j]) / N
  //
  // The target Tensor accumulates no gradient because its requiresGrad
  // was set to false, and the null nextFunctions entry in the subtraction
  // node terminates that branch of the traversal cleanly.

  std::cout << "MSE loss:\n" << mseLoss << "\n";

  mseLoss.backward();

  std::cout << "MSE gradient with respect to input:\n"
            << input.detachGradient() << "\n";

  std::println("-----------------------------------------------------------");

  // CrossEntropyLoss computes the cross-entropy between the predicted
  // probability distributions and a set of one-hot encoded target vectors.
  // It is the standard loss for multi-class classification tasks and is
  // typically paired with a Softmax layer that converts the raw network
  // logits into a proper probability distribution before the loss is applied.
  // For a single sample with C classes, the cross-entropy is
  //
  //     L = -sum_c t_c * log(y_c)
  //
  // where t_c is the one-hot target value for class c, equal to 1 for the
  // correct class and 0 for all others, and y_c is the predicted probability
  // for class c. Because t_c is zero everywhere except the correct class, the
  // sum collapses to the negative log probability assigned to the correct
  // class. Over a batch the calculation applies independently to each sample,
  // and the resulting per-sample losses are collected into a Tensor of shape
  // (batchSize).
  //
  // The implementation mirrors the formula directly using four Tensor
  // operations. First, the elementwise natural logarithm is taken of the
  // input predictions via the log method. Second, the log probabilities are
  // multiplied elementwise with the one-hot target Tensor, zeroing out the
  // contributions from every class that is not the correct one. Third, the
  // result is negated via scalar multiplication by -1. Fourth, reductionSum(1)
  // sums along dimension one, collapsing each row to a single per-sample loss
  // value and producing a result of shape (batchSize).
  //
  // As with MSELoss, every step goes through a standard Tensor operation that
  // installs its own GradFunction and records its operands in savedTensors.
  // The loss Tensor returned by calculateLoss therefore carries a computational
  // graph back through the log, multiply, negate and reductionSum steps and
  // into the input Tensor. If that input Tensor is itself the output of a
  // Softmax layer whose forward built its own four-node subgraph as described
  // in tutorial four, a single backward call on the loss propagates gradients
  // through the CrossEntropyLoss subgraph, through the Softmax subgraph, and
  // all the way back through the network. The gradient accumulated into
  // input[i, c] for each sample i and class c is
  //
  //     dL/d(input[i, c]) = -target[i, c] / input[i, c]
  //
  // which is the derivative of the negative log probability with respect to
  // the predicted probability at that class. When combined with the gradient
  // that flows back through a preceding Softmax, the two contributions
  // simplify algebraically to (predictions - targets), the clean form
  // familiar from logistic regression.
  //
  // The input is expected to contain valid probabilities: all values positive
  // and each row summing to one, as produced by a Softmax layer. Passing
  // values that are zero or negative would produce an undefined result from
  // the logarithm. The target is expected to be one-hot encoded with
  // exactly one non-zero entry per row, and its requiresGrad attribute should
  // be set to false for the same reason as in MSELoss.

  mattTorch::Tensor ceInput({2, 4});
  mattTorch::Tensor ceTarget({2, 4});
  ceTarget.setRequiresGrad(false);

  ceInput[{0, 0}] = 0.1;
  ceInput[{0, 1}] = 0.2;
  ceInput[{0, 2}] = 0.6;
  ceInput[{0, 3}] = 0.1;
  ceTarget[{0, 2}] = 1.0;

  ceInput[{1, 0}] = 0.7;
  ceInput[{1, 1}] = 0.1;
  ceInput[{1, 2}] = 0.1;
  ceInput[{1, 3}] = 0.1;
  ceTarget[{1, 0}] = 1.0;

  std::cout << "ceInput:\n" << ceInput << "\n";
  std::cout << "ceTarget:\n" << ceTarget << "\n";

  // The first sample assigns the highest probability to class 2, which is
  // the correct class, so its loss is -log(0.6) approximately 0.511. The
  // second sample assigns the highest probability to class 0, which is again
  // the correct class, so its loss is -log(0.7) approximately 0.357. The
  // returned Tensor of shape (2) contains both per-sample losses in order.

  mattTorch::criterion::CrossEntropyLoss ce;
  auto ceLoss = ce.calculateLoss(ceInput, ceTarget);

  std::cout << "cross-entropy loss:\n" << ceLoss << "\n";

  ceLoss.backward();

  // The gradient at each element is -target[i, c] / input[i, c]. For the
  // first sample only class 2 has a non-zero target, so only ceInput[0, 2]
  // receives a non-zero gradient of -1.0 / 0.6. For the second sample only
  // class 0 has a non-zero target, so only ceInput[1, 0] receives a non-zero
  // gradient of -1.0 / 0.7. All other elements have a target of zero and so
  // receive a gradient of zero.

  std::cout << "cross-entropy gradient with respect to input:\n"
            << ceInput.detachGradient() << "\n";

  std::println("-----------------------------------------------------------");

  // The Criterion interface integrates cleanly into the training loop
  // established in tutorial six. That loop assembled the loss from raw Tensor
  // operations written directly inside the loop body. Replacing those
  // operations with a single call to criterion.calculateLoss does not change
  // the structure of the loop or the mechanics of the autograd system in any
  // way. The loss Tensor returned by calculateLoss carries exactly the same
  // kind of computational graph regardless of whether it was built by hand
  // inside the loop or constructed by the Criterion, and backward traverses
  // it in exactly the same way in either case.
  //
  // The principal benefit is that the choice of loss function is now
  // factored into a separate object whose construction is a single line at
  // the top of the training setup, parallel in form to the construction of
  // the Network and the Optimiser. Swapping in a different loss function
  // requires changing only that one line rather than rewriting the body of
  // the loop. A Criterion can also be held behind a base-class pointer and
  // passed to a generic training function that calls calculateLoss without
  // knowing which concrete loss is in use, in exactly the same way that a
  // Network iterates through its layers vector without knowing the concrete
  // type of each entry.
  //
  // The example below repeats the training loop from tutorial six using an
  // MSELoss criterion. The canonical order of the five operations within
  // each iteration is unchanged: zeroGrad wipes the gradients from the
  // previous iteration, forward builds the computational graph for the new
  // pass, calculateLoss appends the loss subgraph to its output, backward
  // traverses the combined graph to populate every parameter's gradient slot,
  // and updateParameters applies the SGD step. The computational graph built
  // during an iteration is released when the loss Tensor goes out of scope at
  // the end of the loop body, before the next iteration begins, for the same
  // reason described in tutorial six.

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(4, 8)
                               .addTanhLayer()
                               .addFullyConnectedLayer(8, 4)
                               .build();

  double learningRate{0.05};
  mattTorch::SGD sgd(net.getParameters(), learningRate);
  mattTorch::criterion::MSELoss criterion;

  mattTorch::Tensor trainInput({1, 4});
  mattTorch::Tensor trainTarget({1, 4});
  trainTarget.setRequiresGrad(false);

  for (int j{0}; j < trainInput.getDimensions()[1]; j++) {
    trainInput[{0, j}] = 0.1 * j;
    trainTarget[{0, j}] = 0.05 * j + 0.2;
  }

  for (int iteration{0}; iteration < 50; iteration++) {
    sgd.zeroGrad();

    auto output = net.forward(trainInput);
    auto loss = criterion.calculateLoss(output, trainTarget);

    if (iteration % 5 == 0) {
      std::cout << "iteration " << iteration
                << " loss: " << loss.getData()[0] << "\n";
    }

    loss.backward();
    sgd.updateParameters();
  }

  std::println("-----------------------------------------------------------");

  // The Criterion class completes the picture established across the previous
  // tutorials. The Tensor class is the data primitive, providing aligned
  // contiguous storage and the indexing arithmetic that interprets it as a
  // multidimensional array. The tensor operations are the compute primitives,
  // SIMD-vectorised kernels that combine Tensors elementwise, by matrix
  // multiplication, by broadcasting or by reduction. The autograd system
  // turns each compute primitive into a node of a directed acyclic
  // computational graph, so that a single backward call propagates gradients
  // from any output back to any combination of leaf Tensors that contributed
  // to it. The Layer class packages sequences of tensor operations with
  // learnable parameters into a unit of composition for neural networks. The
  // Network class chains an ordered sequence of layers into a single
  // end-to-end differentiable mapping and exposes the union of their
  // parameters. The Optimiser class consumes the gradients and updates the
  // parameter values in place, ready for the next forward pass. And the
  // Criterion class encapsulates the loss computation behind a clean,
  // reusable interface, appending its own differentiable subgraph to the
  // network's computational graph and thereby connecting the network's
  // output to the gradient signal that drives the entire learning process.

  return 0;
}
