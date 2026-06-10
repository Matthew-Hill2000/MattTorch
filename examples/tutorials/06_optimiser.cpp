#include <mattTorch/mattTorch.h>

#include <iostream>
#include <print>

#include "mattTorch/layer/fullyConnectedLayer/fullyConnectedLayer.h"
#include "mattTorch/network/network.h"
#include "mattTorch/optimiser/sgd/sgd.h"
#include "mattTorch/tensor/tensor/tensor.h"

int main(int argc, char* argv[]) {
  // The previous tutorials have discussed everything that is needed
  // to create a neural network and to compute the gradient of any
  // scalar function of its output with respect to every learnable
  // parameter that contributed to it. The remaining step is the
  // rule that takes those gradients and uses them to update the
  // parameters in a direction that decreases the loss. This is the
  // job of the Optimiser class. An Optimiser holds shared pointers to
  // the parameter Tensors of a network alongside a learning rate that
  // controls the magnitude of each update, and contains an
  // updateParameters method that applies one optimisation step to
  // every parameter using the gradient currently sitting in its
  // gradient attribute. It also has a zeroGrad method that wipes those
  // gradient slots back to zero so that successive training
  // iterations do not accumulate.
  //
  // The Optimiser base class itself is abstract in the same sense
  // that Layer is. Specific optimisation algorithms derive from it
  // and provide the actual update logic from a specific algorithm.
  // The destructor of Optimiser is declared virtual for the same
  // reason that the destructor of Layer is, so that a derived
  // optimiser deleted through an Optimiser* runs its own destructor
  // and cleanly releases its parameter shared pointers as discussed
  // in the previous tutorials.
  //
  // The SGD class is the simplest derivation. in SGD each parameter
  // is updated by subtracting its gradient scaled by the learning rate. The
  // update rule per element is
  //
  //     theta_new = theta_old - learningRate * gradient
  //
  // applied independently to every value of every parameter Tensor.
  // The SGD constructor takes the parameter list returned by
  // Network::getParameters along with a scalar learningRate, and
  // copies both into its own attributes for later use during the
  // training loop.

  std::println("-----------------------------------------------------------");

  // Because the parameter list is a vector of shared pointers to
  // Tensors, the SGD does not own its own private copies of the
  // parameter values. It holds shared pointers that point at the very
  // Tensor objects that the FullyConnectedLayer instances within the network
  // point at, and that the network's own parameter list also points at. Any
  // in-place modification of the underlying values by the optimiser is
  // therefore immediately visible to the layers that use those values
  // during the next forward pas. This is the reason for the shared pointer
  // design decision discussed in tutorials four and five. The optimiser and the
  // network all hold independent shared pointers into the same TensorStorage,
  // the reference count keeps the storage alive while any of them is still in
  // scope, and updates made through any one handle are observed through every
  // other handle.

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(4, 8)
                               .addTanhLayer()
                               .addFullyConnectedLayer(8, 4)
                               .build();

  double learningRate{0.05};
  mattTorch::SGD sgd(net.getParameters(), learningRate);

  std::println("-----------------------------------------------------------");

  // The updateParameters method iterates through the parameter list and applies
  // the SGD update to every parameter Tensor. The update is written using AVX2
  // SIMD intrinsics in the same way as the elementwise kernels described in
  // tutorial two. A single 256-bit register is first filled with four copies of
  // the learning rate via _mm256_set_pd, since every update across every
  // parameter uses the same scalar value. The loop then iterates through the
  // contiguous storage of each parameter four doubles at a time. The current
  // parameter values are loaded into one register with _mm256_load_pd, the
  // current gradient values are loaded into another register with a
  // second _mm256_load_pd, and the three registers are combined with
  // a single _mm256_fnmadd_pd instruction. FNMADD stands for fused
  // negative multiply-add and computes
  //
  //     result = - (a * b) + c
  //
  // across all four lanes in a single fused floating-point operation.
  // This matches the SGD update of c - a * b  with c being the
  // current parameter values, a being the learning rate vector and b
  // being the gradient values. The fused form is more accurate than a
  // separate multiply followed by a subtract because there is only
  // one rounding step instead of two, and on hardware that supports
  // FMA the entire fused operation issues in a single cycle. The
  // updated four parameter values are then written back to contiguous
  // storage with _mm256_store_pd, overwriting the previous values in
  // place. The loop advances by four as long as four full values
  // remain to be processed.
  //
  // The gradient values used here are accessed directly through
  // getGradientData rather than through any Tensor operation, and the
  // new parameter values are written directly into the parameter
  // Tensor's contiguous storage via getData. The optimiser therefore
  // sits entirely outside the computational graph: no GradFunction is
  // installed by the update, no savedTensors are populated, and the
  // graph is unaware that a step has happened. This is exactly what
  // we want, since the parameter update is not something we ever want
  // to differentiate through. Bypassing the Tensor arithmetic operators
  // also means that the update does not allocate any new Tensors or
  // GradFunction nodes, so a training loop of any length performs no
  // heap activity for the optimiser step itself beyond the fixed cost
  // already paid at construction.
  //
  // The zeroGrad method iterates through the parameter list a second time and
  // for each parameter Tensor calls getGradientData to obtain the pointer to
  // the contiguous storage of its gradient Tensor, then calls std::fill to set
  // every value in that storage to zero. The reason this step is necessary,
  // rather than the gradient slot simply being overwritten by the next backward
  // pass, is the GradAccumulator's behaviour described in tutorial three. When
  // the backward pass reaches a leaf, its GradAccumulator adds the incoming
  // gradient to whatever is already sitting in the leaf's gradient Tensor, it
  // does not replace it. This is the behaviour that allows a single leaf to
  // receive contributions from multiple paths through the graph during one
  // backward pass, but it also means that without an explicit reset between
  // iterations, the gradients from prior iterations would persist and corrupt
  // the step direction of the next one. zeroGrad is the reset that
  // prepares the gradient slots for the next backward. Tutorial three's
  // detachGradient method achieves the same effect for a single leaf by
  // swapping in a fresh zero Tensor and returning the previous one; zeroGrad is
  // the  equivalent that resets every parameter in the network without
  // producing a return value.

  std::println("-----------------------------------------------------------");

  // The canonical training loop ties everything together. Each
  // iteration calls zeroGrad to wipe the previous iteration's
  // gradients, runs the forward pass through the network to produce
  // an output Tensor with a full computational graph attached, uses
  // any chosen scalar function of that output as a loss, calls
  // backward on the loss to populate every parameter's gradient slot,
  // and finally calls updateParameters to step every parameter
  // against its newly populated gradient. Here we use a simple
  // mean squared error using only the Tensor operations from tutorial
  // two. A subtraction to form the difference between prediction and
  // target, an elementwise multiplication of the difference by itself
  // to square it, and a mean to collapse the result to a single
  // scalar. Each of these operations adds its own GradFunction as
  // described in tutorial three, so the loss Tensor at the end of
  // the chain carries the full backward graph all the way back
  // through the squared and difference intermediates, through every
  // layer of the network, and finally into the weight and bias
  // GradAccumulators of every FullyConnectedLayer.
  //
  // The target Tensor has its requiresGrad attribute set to false, since we
  // never want to compute gradients with respect to the ground truth. As
  // described in tutorial three, the subtraction operator will see one operand
  // requesting gradients (the network output) and one not (the target), and
  // will add a GradFunction node whose nextFunctions for the non-grad operand
  // is simply null, terminating that branch of the backward traversal cleanly
  // at the target. No inputGradient is passed explicitly to backward, so
  // Tensor::backward seeds the traversal with the all-ones Tensor described in
  // tutorial three. For a scalar loss this isthe dL/dL = 1 seed that
  // produces the ordinary gradient of the loss with respect to every parameter.

  mattTorch::Tensor input({4, 4});
  mattTorch::Tensor target({4, 4});
  for (int i{0}; i < 4; i++) {
    for (int j{0}; j < 4; j++) {
      input[{i, j}] = 0.1 * (i + j);
      target[{i, j}] = 0.05 * (i - j) + 0.2;
    }
  }
  target.setRequiresGrad(false);

  for (int iteration{0}; iteration < 50; iteration++) {
    sgd.zeroGrad();

    auto output = net.forward(input);
    auto difference = output - target;
    auto squared = difference * difference;
    auto loss = squared.mean();

    if (iteration % 5 == 0) {
      std::cout << "iteration " << iteration << " loss: " << loss.getData()[0]
                << "\n";
    }

    loss.backward();
    sgd.updateParameters();
  }

  std::println("-----------------------------------------------------------");

  // A few details about the loop above are worth emphasising. The
  // lifetime of the entire computational graph built during the
  // iteration is bounded by the scope of the loss Tensor and the
  // intermediate Tensors output, difference and squared. As the
  // for-loop body ends, those Tensors go out of scope, their
  // gradFunction shared pointers drop to zero, and the cascade
  // described in tutorial three releases every GradFunction in the
  // graph and every retained TensorStorage that contributed to it
  // before the next iteration begins. The only state that persists
  // across iterations is the parameter values themselves, updated in
  // place by the optimiser, and the gradient slots that zeroGrad will
  // wipe at the top of the next pass.
  //
  // The order of zeroGrad, forward, backward and updateParameters
  // matters. zeroGrad must run before backward, otherwise the
  // gradient produced by backward would be added on top of stale
  // values left over from the previous iteration. updateParameters
  // must run after backward, otherwise it would step in the direction
  // of those same stale gradients before the new ones have a chance
  // to be computed. The forward pass must come before backward for
  // the obvious reason that backward needs a graph to traverse, and
  // the loss must come from a Tensor produced by that forward pass
  // for the gradients to flow back into the network's parameters at
  // all. Any order respecting these dependencies is valid; the
  // conventional zeroGrad, forward, backward, updateParameters
  // sequence is simply the clearest reading of the same dependency
  // graph.
  //
  // The picture that emerges across the six tutorials is the
  // following. The Tensor class from tutorial one is the data
  // primitive, providing aligned contiguous storage and the indexing
  // arithmetic that interprets it as a multidimensional array. The
  // tensor operations from tutorial two are the compute primitives,
  // SIMD-vectorised kernels that combine Tensors elementwise, by
  // matrix multiplication, by broadcasting or by reduction. The
  // autograd system from tutorial three is the bookkeeping that
  // turns each compute primitive into a node of a directed acyclic
  // computational graph, so that a single backward call propagates
  // gradients from any output back to any combination of leaf
  // Tensors that contributed to it. The Layer class from tutorial
  // four is the unit of composition for neural networks, packaging a
  // configurable sequence of tensor operations along with the
  // learnable parameter Tensors that participate in them. The
  // Network class from tutorial five is the orchestrator, chaining
  // an ordered sequence of layers into a single end-to-end
  // differentiable mapping and exposing the union of their
  // parameters. And the Optimiser class from this tutorial closes
  // the loop by consuming the gradients produced by the autograd
  // system and updating the parameter values in place, ready for the
  // next forward pass to consume them.

  return 0;
}
