
#include <mattTorch/mattTorch.h>

#include <iostream>
#include <print>

#include "mattTorch/tensor/tensor/tensor.h"

int main(int argc, char* argv[]) {
  // An important functionality of the mattTorch library, underpinning
  // the machine learning process, is the ability to calculate the gradients of
  // tensors with respect to the Tensors from which they were generated as a
  // result of any combination of the defined mathematical operations.
  //
  // Neural networks operate as a series of parameters that consecutively
  // operate on an input Tensor in order to produce an output tensor
  // that represents the prediction of the network. To improve the network
  // during the training process the gradient of the loss computed from the
  // prediction and true value needs to be calculated with respect to each of
  // the parameters. Stochastic gradient descent or a similar algorithm can then
  // be used to update each of the parameters.
  //
  // To facilitate this process for any combination of layers that utilise any
  // combination of mathematical operations, an automatic differentiation system
  // is built on top of the Tensor class that allows the gradients to be
  // computed automatically after any arbitrary number of mathematical
  // operations. Specifically, it is reverse mode automatic differentiation,
  // meaning the gradients of a single output can be computed with respect to
  // all of the inputs with a single backwards pass.
  //
  // Every operation performed on a Tensor records both the inputs and a
  // function that knows how to differentiate the result with respect to the
  // inputs. Together these form a directed graph from any output Tensor back to
  // the leaf Tensors from which it was built. Calling backward on the output
  // then walks this graph in reverse, applying the chain rule at each node, and
  // accumulates the final gradients into the gradient attribute of each leaf.
  // The rest of this tutorial works through how this is set up and executed
  // within the Tensor class.
  //
  // Focussing first on a single operation, we look at how the gradients are
  // calculated after a single elementwise multiplication between two tensors a
  // and b.
  //
  mattTorch::Tensor a({3, 3});
  mattTorch::Tensor b({3, 3});

  for (int i{0}; i < a.getDimensions()[0]; i++) {
    for (int j{0}; j < a.getDimensions()[1]; j++) {
      a[{i, j}] = 25 - i * j;
      b[{i, j}] = 4 + i + j;
    }
  }

  std::println("Tensor a: \n");
  std::cout << a << "\n";

  std::println("Tensor b: \n");
  std::cout << b << "\n";

  // When you create a tensor in this way, the values, dimensions, strides etc
  // are all set up just as previously described. In addition to this set up,
  // and not previously discussed, is the set up required for future gradient
  // calculations. A tensor created from scratch using any of the defined Tensor
  // constructors is known as a leaf tensor, the reason for which will become
  // clear soon. In the constructor for a leaf Tensor a shared pointer to a
  // separate Tensor object used to store the gradient values is allocated
  // as the gradient attribute. This gradient storing tensor is constructed with
  // the isLeaf parameter set to false to ensure an endless recursive call to
  // the constructor isn't initiated.
  //
  // The tensor is also initialised with the gradFunction shared pointer to a
  // GradAccumulator object, a special child of the general GradFunction class.
  // The GradAccumulator class constructor takes the shared pointer to the
  // gradient Tensor and the dimensions of the Tensor as parameters, and is
  // utilised later on to redirect the computed values of the gradient into the
  // Tensor's gradient Tensor.

  // Multiplying the two tensors together with elementwise multiplication
  // produces the result in exactly the same way as described in the previous
  // tutorial

  auto c = a * b;

  std::println("c = a*b: \n");
  std::cout << c << "\n";

  // However we can now discuss the rest of the implementation within the
  // * overload that serves to set the stage for the automatic differentiation.
  // The same pattern is followed by every mathematical operation defined for
  // the Tensor class. After the kernel has calculated and stored the result,
  // an if statement fires if both operand Tensors have their
  // gradData.requiresGrad attribute set to false, in which case no further
  // gradient logistics are performed.
  //
  // Otherwise, both of the operand Tensors are shallow copied
  // into a std::vector<Tensor> container called savedTensors, and the pointers
  // to the GradFunction objects of both operand Tensors are copied into a
  // vector called nextFunctions. These two vectors are used in the construction
  // of a new GradFunction object that is created via a shared pointer which is
  // set to be the gradFunction pointer attribute of the result Tensor. Each
  // mathematical operation defined for the Tensor class has an associated
  // special GradFunction class, derived from the base GradFunction class. Each
  // of these derived classes overloads the backward method of the base
  // GradFunction class with the appropriate calculations necessary to compute
  // the derivative of the result tensor with respect to those tensors that
  // participated in its creation. The savedTensors and nextFunctions vectors
  // are passed to the GradFunction during its construction such that it has
  // access to the values required to calculate the gradients and the pointers
  // to the Tensors to which it should pass those gradients.
  //
  // For the case of the elementwise multiplication above, the result tensor c
  // is allocated a pointer to a GradMultiply object. The savedTensors vector
  // contains both a and b, and the nextFunctions vector contains shared
  // pointers to the GradAccumulator objects of a and b. Drawing the
  // computational graph as the graph of GradFunction objects connected by
  // nextFunctions pointers, this produces:
  //
  //     c.gradFunction (GradMultiply)
  //                  |
  //                  |--> a.gradFunction (GradAccumulator)
  //                  |
  //                  `--> b.gradFunction (GradAccumulator)
  //
  //     c.gradFunction.savedTensors  = {a, b}
  //     c.gradFunction.nextFunctions = {a.gradFunction, b.gradFunction}
  //
  // The arrows point in the direction the backward pass will later traverse:
  // from the GradFunction of the output down through each branch of
  // nextFunctions to the GradAccumulators of the leaves.
  //
  // Calling c.backward() begins the traversal of this graph from the
  // GradMultiply object of c to the GradAccumulator objects of a and b. The
  // backward method of c calls the backward method of its GradMultiply
  // attribute. The backward method of the GradMultiply object then calculates
  // the gradients dc/da = b and dc/db = a. Each of these gradients is in turn
  // passed as the inputGradient parameter to the backward method of the
  // corresponding GradFunction stored in nextFunctions, which in this case
  // are the GradAccumulator objects of a and b. The backward method of a
  // GradAccumulator simply adds the gradient it receives into the gradient
  // Tensor that it holds a shared pointer to. So once c.backward() has run,
  // a.gradient contains dc/da and b.gradient contains dc/db. The use of
  // addition here is what gives the GradAccumulator its name, and becomes
  // important as soon as the same leaf Tensor participates in more than one
  // path through the graph.
  //
  // One detail glossed over so far is where the very first inputGradient
  // comes from. Before any of the GradFunctions are invoked, Tensor::backward
  // constructs an inputGradient Tensor with the same dimensions as c and
  // fills it with the value 1.0. This Tensor is the seed that is handed to
  // c.gradFunction.backward and kicks off the chain rule traversal. The
  // choice of all ones corresponds to choosing dL/dc = 1 for every element
  // of c. For a scalar output this is simply the derivative of the loss
  // with respect to itself, and the gradients accumulated into the leaves
  // are the usual gradients of the loss with respect to the parameters.
  // For a non-scalar output the same procedure produces the vector-Jacobian
  // product 1^T J between the row vector of ones and the Jacobian of c
  // with respect to the leaves, which is equivalent to taking the gradient
  // of the sum of c with respect to the leaves.
  //
  // The two-argument overload Tensor::backward(Tensor& inputGradient, bool
  // higherDerivative) lets you replace the all-ones seed with a Tensor of
  // your own, in which case the same traversal produces the vector-Jacobian
  // product v^T J between your seed v and the Jacobian. Providing a one-hot
  // Tensor picks out the gradient of a single element of c, providing a
  // unit vector along a direction of interest gives the directional
  // derivative along that direction, and providing an arbitrary Tensor
  // gives the contraction of the full Jacobian against it. This is the
  // primitive on top of which every higher-level gradient computation in
  // the library is built.
  //
  // Once the backward pass has populated the leaves, the gradients are
  // accessed via the detachGradient method. detachGradient does two things
  // in a single call: it returns the Tensor currently sitting in the leaf's
  // gradient slot, and it installs a fresh zero Tensor of the same
  // dimensions in its place, rewiring the leaf's GradAccumulator via shared
  // pointer to point at the new gradient Tensor. This means subsequent
  // backward calls accumulate into a clean slot rather than on top of the
  // value that was just returned, and allows the same leaf to participate
  // in multiple independent backward passes without manual bookkeeping. For
  // the case where the gradient is no longer needed but the slot should
  // still be cleared, the resetGradient method performs the same reset
  // without returning the previous value, and serves as the analogue of the
  // zero_grad step that typically precedes each optimiser step.

  c.backward();

  std::cout << a.detachGradient() << "\n";
  std::cout << b.detachGradient() << "\n";

  // To see how this scales to longer computational graphs, consider extending
  // the example by introducing a third leaf Tensor d and computing
  //
  //     e = c * d
  //
  // Construction of d follows exactly the same procedure as a and b,
  // allocating its own gradient Tensor and a GradAccumulator as its
  // gradFunction. The multiplication c * d goes through the same * overload
  // as before, so the same gradient bookkeeping is performed for e. The
  // savedTensors vector of e's GradFunction contains c and d, and the
  // nextFunctions vector contains the GradFunction pointers of c and d.
  // Importantly, the GradFunction pointer of c is no longer a
  // GradAccumulator, as c is not a leaf Tensor, it is the GradMultiply object
  // that was created during the calculation of c = a * b. The graph now
  // looks like:
  //
  //     e.gradFunction (GradMultiply)
  //                  |
  //                  |--> c.gradFunction (GradMultiply)
  //                  |                  |
  //                  |                  |--> a.gradFunction (GradAccumulator)
  //                  |                  |
  //                  |                  `--> b.gradFunction (GradAccumulator)
  //                  |
  //                  `--> d.gradFunction (GradAccumulator)
  //
  //     e.gradFunction.savedTensors  = {c, d}
  //     e.gradFunction.nextFunctions = {c.gradFunction, d.gradFunction}
  //
  // where the GradAccumulator objects of a, b and d still hang off the leaf
  // Tensors as before, and the inner c.gradFunction is the same GradMultiply
  // node that was constructed during c = a * b.
  //
  // Calling e.backward() now requires the chain rule to propagate gradients
  // all the way back to a and b. The backward method of e's GradMultiply
  // object calculates de/dc = d and de/dd = c, and passes each of these to
  // the backward method of the corresponding GradFunction in nextFunctions.
  // The gradient de/dd is passed to the GradAccumulator of d, which adds it
  // directly into d.gradient as in the single-operation case. The gradient
  // de/dc however is passed to the GradMultiply object of c, as the
  // inputGradient parameter of its backward method, rather than to a
  // GradAccumulator. This is where the chain rule is applied. Instead of
  // producing dc/da and dc/db as before, this backward call uses the
  // inputGradient to produce
  //
  //     de/da = de/dc * dc/da = d * b
  //     de/db = de/dc * dc/db = d * a
  //
  // which are then passed on to the GradAccumulator objects of a and b in
  // exactly the same way as in the single-operation case. The same pattern
  // continues for graphs of arbitrary depth: at each non-leaf node, the
  // backward method multiplies its incoming inputGradient by the local
  // derivatives of that operation, and forwards the result on to the
  // GradFunctions of its operands, until the chain eventually terminates at
  // the GradAccumulators of the leaf Tensors.
  //
  // Worth noting at this point is that only leaf Tensors have a slot in
  // which to retain their gradients. Intermediate Tensors like c, that are
  // the result of operations rather than leaves of the graph, propagate
  // gradients during the backward pass but do not store them anywhere
  // accessible. The GradMultiply object that c carries does not own a
  // gradient Tensor of its own, only the GradAccumulator of a leaf does, so
  // de/dc is computed and forwarded but is not available afterwards. If the
  // gradient of an intermediate Tensor is needed, the two-argument backward
  // overload can be used by manually seeding a backward call at that
  // intermediate, in which case its gradient with respect to the leaves
  // becomes the seed and the contribution from later operations is treated
  // as already factored into it.
  //
  // The other detail worth mentioning is what happens at the boundary of
  // the graph, where an operation has an operand whose gradFunction is
  // null. This occurs whenever an intermediate Tensor was the result of an
  // operation in which neither operand required grad, since the * overload
  // only allocates a GradFunction for its result when at least one operand
  // requires it. If such a Tensor later participates in an operation that
  // does require gradient bookkeeping, the null gradFunction is pushed into
  // the new GradFunction's nextFunctions vector as-is. Every derived
  // backward method checks each entry of nextFunctions against null before
  // recursing into it, so a null entry simply terminates that branch of
  // the traversal without complaint. This is what allows mixed-grad graphs,
  // where only some of the leaves require gradients, to be computed
  // correctly without separate handling at every operation.

  mattTorch::Tensor d({3, 3});

  for (int i{0}; i < d.getDimensions()[0]; i++) {
    for (int j{0}; j < d.getDimensions()[1]; j++) {
      d[{i, j}] = 2 + i - j;
    }
  }

  auto e = c * d;

  std::println("e = c*d: \n");
  std::cout << e << "\n";

  e.backward();

  std::cout << a.detachGradient() << "\n";
  std::cout << b.detachGradient() << "\n";
  std::cout << d.detachGradient() << "\n";

  // Computing higher-order derivatives, such as the entries of a Hessian
  // matrix, requires being able to differentiate a gradient that was itself
  // produced by an earlier backward pass. This is what the higherDerivative
  // parameter of the backward method controls, by deciding whether or not a
  // new computational graph is built for the gradient Tensors themselves
  // during the backward pass.
  //
  // When backward is called with higherDerivative set to false, which is the
  // default value, every operation that runs inside a GradFunction's
  // backward method is performed without any gradient logistics. This is
  // achieved at the start of each derived backward method by setting
  // requiresGrad to false on its savedTensors, so that the * overload, and
  // every other operation, does not allocate a new GradFunction node for the
  // result. The gradient Tensors produced by the backward pass are therefore
  // plain Tensors with no history attached, and they cannot themselves be
  // backpropagated through.
  //
  // When backward is called with higherDerivative set to true the savedTensors
  // keep their requiresGrad attribute set to true, and the inputGradient that
  // Tensor::backward constructs is also marked as requiring grad before being
  // passed in. As a result, every multiplication, addition or other operation
  // performed inside the backward method goes through the same gradient
  // logistics that was described in the single-operation case earlier,
  // allocating a new GradFunction object for each intermediate gradient Tensor.
  // This produces a second computational graph, distinct from the original
  // forward graph, that encodes how each gradient was computed from the saved
  // Tensors.
  //
  // The GradAccumulator's backward method also branches on this flag. With
  // higherDerivative set to true, instead of just copying or adding the
  // values of inputGradient into the leaf's gradient Tensor, it additionally
  // sets the gradFunction of the gradient Tensor to the gradFunction of the
  // inputGradient and marks the gradient Tensor as requiring grad. The
  // leaf's gradient Tensor is therefore no longer a leaf itself, it is now a
  // non-leaf node in the gradient graph, with its own backward chain
  // reaching back through the operations that produced it.
  //
  // Concretely, for the original c = a * b example, calling
  //
  //     c.backward(true);
  //
  // walks the same path through the original forward graph
  //
  //     c.gradFunction (GradMultiply)
  //                  |
  //                  |--> a.gradFunction (GradAccumulator)
  //                  |
  //                  `--> b.gradFunction (GradAccumulator)
  //
  // but now the multiplications b * inputGradient and a * inputGradient
  // inside GradMultiply::backward each build their own GradMultiply object
  // as part of a new gradient graph. Calling these GradMultiply_A and
  // GradMultiply_B, this gradient graph looks like
  //
  //     a.gradient.gradFunction (GradMultiply_A)
  //                  |
  //                  |--> b.gradFunction (GradAccumulator)
  //                  |
  //                  `--> inputGradient.gradFunction (GradAccumulator)
  //
  //     b.gradient.gradFunction (GradMultiply_B)
  //                  |
  //                  |--> a.gradFunction (GradAccumulator)
  //                  |
  //                  `--> inputGradient.gradFunction (GradAccumulator)
  //
  //     a.gradient.gradFunction.savedTensors  = {b, inputGradient}
  //     a.gradient.gradFunction.nextFunctions = {b.gradFunction,
  //                                              inputGradient.gradFunction}
  //     b.gradient.gradFunction.savedTensors  = {a, inputGradient}
  //     b.gradient.gradFunction.nextFunctions = {a.gradFunction,
  //                                              inputGradient.gradFunction}
  //
  // The two gradient graphs sit alongside the original forward graph and
  // share nodes with it: GradMultiply_A's nextFunctions include
  // b.gradFunction, which is the very same GradAccumulator that already
  // appeared in c.gradFunction's nextFunctions, and similarly
  // GradMultiply_B's nextFunctions include a.gradFunction. The link from
  // a.gradFunction in the forward graph to a.gradient.gradFunction in the
  // gradient graph is not a nextFunctions pointer, it is the side effect
  // performed by the GradAccumulator's backward method when higherDerivative
  // is true: writing the inputGradient's gradFunction into the gradient
  // Tensor that the GradAccumulator holds.
  //
  // The Tensor stored as a.gradient still holds the values of b, but it now
  // carries a gradFunction that points back to b and inputGradient through
  // GradMultiply_A. Detaching it with a.detachGradient and calling backward
  // on the result then traverses this branch of the gradient graph in
  // exactly the same way as any other backward pass:
  //
  //     a.gradient.gradFunction (GradMultiply_A)
  //                  |
  //                  |--> b.gradFunction --> b.gradient
  //                  |
  //                  `--> inputGradient.gradFunction
  //
  // producing d(dc/da)/db = 1 into b.gradient via b's GradAccumulator. The
  // gradient sent to inputGradient's GradAccumulator is the value b, but
  // since inputGradient is internal to c.backward and not exposed, this
  // contribution is simply discarded.
  //
  // The same mechanism extends recursively: passing higherDerivative = true
  // to the second backward call would itself build a third graph on top of
  // a.gradient.gradFunction, allowing a third derivative to be taken, and so
  // on for as many orders of differentiation as required.

  auto h = a * b;
  h.backward(true);

  // Detaching dh/da from a returns the gradient Tensor along with its newly
  // built gradient graph. Calling detachGradient on b at this point simply
  // discards the first-order gradient dh/db that was just accumulated into
  // it, so that the second backward pass below writes the pure second
  // derivative into a fresh gradient Tensor.
  mattTorch::Tensor dh_da = a.detachGradient();
  b.detachGradient();

  dh_da.backward();

  std::println("d(dh/da)/db: \n");
  std::cout << b.detachGradient() << "\n";

  // With the mechanics of the forward and backward passes established, it is
  // worth understanding the lifetime of the computational graph itself,
  // since the entire automatic differentiation system rests on every node of
  // that graph being reachable at the moment backward is called and freed
  // again once it is no longer needed. The graph is not a single owned object
  // sitting somewhere central, it is a collection of GradFunction objects on
  // the heap whose lifetimes are managed entirely by std::shared_ptr.
  //
  // Each GradFunction lives behind a shared pointer. The result Tensor of an
  // operation holds the shared pointer to its own GradFunction in its
  // gradFunction attribute, and that GradFunction in turn holds shared
  // pointers to the GradFunctions of its operands in its nextFunctions
  // vector. The arrows of nextFunctions therefore double as the ownership
  // edges of the graph, pointing from a child node down to its parents. A
  // GradFunction is kept alive by the sum of all shared pointers that
  // currently reference it, which is one for each child operation that
  // consumed the corresponding Tensor as an operand, plus one for the Tensor
  // itself if that Tensor is still in scope. The GradAccumulators of leaf
  // Tensors are an additional case, they are held by the leaf Tensor's
  // gradFunction attribute itself, so they live for as long as their leaf
  // does and outlast the rest of the graph naturally.
  //
  // In the e = c * d example from earlier, the GradMultiply node attached to
  // e is held alive by e.gradFunction. Its nextFunctions vector holds
  // shared pointers to c.gradFunction and d.gradFunction, which keeps the
  // inner GradMultiply node attached to c and the GradAccumulator attached
  // to d alive even if c and d themselves were to go out of scope, because
  // the e node still references them. The savedTensors vector of each
  // GradFunction contains shallow copies of the operand Tensors, which is to
  // say Tensor objects that share their underlying TensorStorage shared
  // pointer with the originals. The data needed to compute the derivatives
  // during the backward pass is therefore kept alive through the entire
  // graph as long as any GradFunction that saved it remains alive.
  //
  // When the result Tensor at the top of a graph falls out of scope, its
  // destructor releases its shared pointer to its gradFunction. If no other
  // shared pointer references that GradFunction, its reference count reaches
  // zero and it is destroyed. Its destruction releases the shared pointers
  // in its nextFunctions vector, which may in turn drop the reference count
  // of the parent GradFunctions to zero and trigger their destruction, and
  // so the deallocation cascades down the graph in topological order. Each
  // GradFunction's destruction also releases its savedTensors, which drops
  // the reference count on the shared TensorStorage objects that held the
  // forward-pass values, and these are freed too once nothing else holds
  // them. The cascade terminates at the GradAccumulators of the leaves,
  // which are kept alive by their leaf Tensors and so survive until the
  // leaves themselves are destroyed.
  //
  // The reason this scheme is safe against memory leaks is structural rather
  // than incidental. Every owning edge in the graph is a shared pointer
  // pointing from a child node to a parent, and there are no edges pointing
  // the other way. A GradFunction does not know about and does not hold any
  // reference to the GradFunctions of operations that consumed its output,
  // it only knows about the GradFunctions of the operands that produced its
  // input. The graph is therefore a directed acyclic structure in which all
  // shared pointer edges flow in the same direction, which makes reference
  // counting cycles impossible by construction. Repeated use of the same
  // leaf Tensor in multiple operations introduces multiple shared pointers
  // into the same GradAccumulator from different downstream nodes, but each
  // of those is still a child-to-parent edge, so no cycle is formed. There
  // are no weak pointers, no manual new and delete, and no raw owning
  // pointers anywhere in the graph, so once the head Tensor of the graph is
  // released the entire structure is reclaimed deterministically as the
  // destructors run.
  //
  // The cases that do require care all involve unintentionally extending
  // the lifetime of a graph past the point at which it is wanted. The most
  // common is simply holding on to the result Tensor of a long sequence of
  // operations longer than necessary, which keeps the head of the graph
  // alive and through it every GradFunction and every saved TensorStorage
  // that contributed to its construction. In a training loop this is what
  // would otherwise allow successive iterations to accrete memory if the
  // result Tensor of each forward pass were stored rather than overwritten,
  // since each iteration's graph would remain pinned by its stored head.
  // Overwriting the result Tensor at the end of each iteration is sufficient
  // to release the previous graph, the destructor of the discarded Tensor
  // drops the head shared pointer and the cascade described above frees
  // everything that was reachable from it.
  //
  // The in-place operators deserve specific mention because their effect on
  // graph lifetime is less obvious than the non-mutating ones. Operators
  // such as operator+= and operator*= overwrite the gradFunction attribute
  // of the receiving Tensor with a new GradFunction whose savedTensors
  // contain a copy of the pre-operation state. The previous gradFunction is
  // released by this overwrite, but its replacement now holds a savedTensor
  // referencing the storage that the receiver had before the operation, so
  // the memory associated with the pre-operation state is retained for as
  // long as the new gradFunction lives. A long sequence of in-place updates
  // therefore does not free its own history, each step pushes the previous
  // step into a savedTensor of the new node, and the chain of saved states
  // is only released when the receiving Tensor's final gradFunction is
  // released.
  //
  // Higher-order differentiation requires the same care applied to a second
  // graph. When backward is called with higherDerivative set to true the
  // GradAccumulator of each leaf writes the gradFunction of the inputGradient
  // into the leaf's gradient Tensor as part of its side effect, which means
  // the gradient Tensor now holds the head of an entirely new gradient
  // graph. That gradient graph contains its own savedTensors referencing
  // both the forward-pass Tensors and the inputGradient Tensor that was
  // constructed inside Tensor::backward, so its memory footprint is
  // comparable to that of the forward graph it was built from. Detaching
  // the gradient from the leaf with detachGradient transfers the gradient
  // graph into the returned Tensor, so the graph remains alive for as long
  // as that returned Tensor is held. Letting the returned Tensor go out of
  // scope releases the gradient graph in the same cascading manner as a
  // forward graph. If the same leaf is used as input to several
  // higher-derivative backward passes without the previous returned
  // gradient Tensor being discarded, each gradient graph remains alive
  // alongside the others, and the memory cost grows accordingly.
  //
  // The actual size of each GradFunction node is small. Each node carries a
  // savedTensors vector containing a handful of Tensor objects, each Tensor
  // object itself being a thin wrapper that holds shared pointers and a few
  // pieces of metadata such as dimensions, strides and an offset. The
  // nextFunctions vector contains a handful of shared pointers to other
  // GradFunctions, and the node may additionally hold one or two scalars
  // depending on the operation. The dominant contribution to memory is
  // therefore not the GradFunction objects themselves but the underlying
  // TensorStorage blocks that the savedTensors keep alive, which contain
  // the forward-pass values needed for the backward pass and scale with the
  // size of the operands. For a forward computation consisting of N
  // operations on tensors of size S, the GradFunction headers account for
  // O(N) memory in total, and the retained forward-pass storage accounts
  // for O(N * S) memory in the worst case where each operation needs to
  // keep its own copy of an operand of size S. Operations that do not need
  // an operand for their backward computation simply do not save it, so
  // operations like addition contribute only the GradFunction header,
  // whereas multiplication and division retain the operand storage they
  // depend on.
  //
  // The time complexity of the forward pass for a sequence of N operations
  // is O(N * W), where W is the per-operation kernel cost and scales with
  // the size of the tensors involved. The gradient logistics adds an
  // O(1) overhead per operation for constructing the GradFunction and
  // copying the shared pointers into savedTensors and nextFunctions, which
  // is dominated by the kernel cost in any realistic computation. The
  // backward pass visits each GradFunction in the graph exactly once and
  // performs a number of operations per node that mirrors the cost of the
  // forward operation it differentiates, so its time complexity is also
  // O(N * W). Forward and backward together are therefore linear in the
  // length of the computational graph.
  //
  // Higher-order derivatives change this picture because each successive
  // order builds and traverses an additional graph. The backward method of
  // each GradFunction performs a small constant number of operations,
  // typically one or two multiplications or additions per operand, so when
  // higherDerivative is true the gradient graph constructed from a forward
  // graph of N nodes itself contains O(N) nodes, multiplied by a small
  // branching constant that depends on the operations involved. Taking a
  // second derivative through that gradient graph builds a third graph of
  // size that scales again by the same constant per node visited, and in
  // general the kth-order graph has size that grows geometrically in k
  // while remaining linear in N. The cumulative memory cost of holding all
  // the graphs in flight while computing a kth-order derivative is
  // therefore O(N * c^k * S), and the time complexity of producing the
  // kth-order derivative is O(N * c^k * W), where c is the per-node
  // branching constant of the operations in the graph. Practical use of
  // second and third derivatives is well within reach, but pushing to
  // arbitrarily high orders is bounded by this geometric growth rather
  // than by any architectural limit of the library.
}
