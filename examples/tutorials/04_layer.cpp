

#include <mattTorch/mattTorch.h>

#include <iostream>
#include <print>

#include "mattTorch/layer/ReLU/ReLU.h"
#include "mattTorch/layer/fullyConnectedLayer/fullyConnectedLayer.h"
#include "mattTorch/layer/softmax/softmax.h"
#include "mattTorch/layer/tanh/tanh.h"
#include "mattTorch/tensor/tensor/tensor.h"

int main(int argc, char* argv[]) {
  // Neural networks are typically constructed from a number of distinct
  // layers. Each layer takes a Tensor input, performs some form of
  // mathematical manipulation on that Tensor, and produces an output
  // Tensor. Networks chain such layers together so that the output from
  // one layer serves as the input to the next. The original input to the
  // first layer is ultimately transformed, layer after layer, into the
  // final output of the network.
  //
  // mattTorch has a Layer class, allowing you to define a layer made up
  // of any combination of the defined mathematical operations. In order
  // to create a layer, you must define a class that inherits from the
  // base Layer class. Your derived class should contain whatever
  // learnable parameters it requires as attributes, and must then
  // override: the forward method, which defines the mathematical
  // operations that transform the input Tensor into the output Tensor;
  // the getParameters method, which returns a vector of shared pointers
  // to those parameters that you expect the machine learning process to
  // fine tune; the getNumParameters method, which returns the total
  // number of such parameters within the Layer; and of course the
  // constructor, which initialises the Layer and assigns starting values
  // to the parameters.
  //
  // The Layer base class itself is abstract. The forward, getParameters
  // and getNumParameters methods are all declared pure virtual, which
  // prevents Layer itself from ever being instantiated and forces every
  // derived layer to provide its own implementation. The destructor is
  // declared virtual so that a derived layer destroyed through a base
  // class pointer runs the correct destructor and properly  releases its
  // parameter Tensors. This matters because the Network class introduced
  // in the next tutorial holds its layers as a std::vector<Layer*> and
  // iterates through that vector during the forward pass without knowing the
  // concrete type of each entry. Polymorphism through the Layer pointer
  // is what lets a single Network::forward implementation call a
  // FullyConnectedLayer, a ReLU and a Softmax in sequence without ever
  // naming them explicitly.
  //
  // The getParameters method returns a std::vector<std::shared_ptr<Tensor>>
  // rather than references or raw pointers for two reasons. The first is
  // that the optimiser introduced in a later tutorial holds onto these
  // pointers across the lifetime of training and needs the underlying
  // Tensors to remain alive. The second is that the same parameter
  // Tensor  appears in two places at once: as a leaf participating in the
  // computational graph through its GradAccumulator, and as an entry in the
  // optimiser's parameter list to be updated in place. The shared pointer is
  // what lets both views point at the same underlying TensorStorage without
  // either side owning it exclusively, and is also what allows getParameters to
  // be called repeatedly without the parameter Tensors being copied into the
  // returned vector.
  //
  // The forward method of a Layer is the point at which the layer plugs
  // itself into the automatic differentiation system described in the
  // previous tutorial. Every mathematical operation called inside forward
  // is a normal Tensor operation, which means it allocates a GradFunction
  // on its result and stores the operands into the result's savedTensors
  // and nextFunctions vectors in exactly the way described in tutorial
  // three. The Tensor returned by forward therefore carries with it the
  // entire graph that produced it, with the layer's learnable
  // parameter Tensors sitting as leaves of that graph. connecting
  // layers is therefore just connecting operations, and the graph of a
  // multi-layer network is the concatenation of the per-layer subgraphs,
  // joined together by the fact that the output Tensor of one layer is
  // passed as the input Tensor of the next.

  std::println("-----------------------------------------------------------");

  // The canonical example of a layer used in neural networks is that
  // which forms the fully connected network. A fully connected layer
  // takes an input of shape (batchSize x inputs), multiplies it by a
  // matrix of parameters called the weights of shape (inputs x outputs)
  // to produce an intermediate result of shape (batchSize x outputs),
  // and finally adds a vector of parameters called the bias of shape
  // (outputs) to each row of that result. It is typically helpful to
  // think of this process visually, where each element of an input row
  // is connected to each element of the corresponding output row. For
  // each output, each input is multiplied by a specific scalar weight
  // value, the products are summed, and then a scalar bias is added.
  // This is represented with lines that connect each input to each
  // output. The outputs of this layer then serve as the inputs to the
  // next layer, with each layer's outputs being connected to the next
  // layer's outputs by a fresh set of weights and biases.
  //
  // The batch dimension is worth talking about. Although it is natural
  // to describe a fully connected layer as acting on a single vector of
  // length N, in practice we want to push many independent samples
  // through the same layer in a single forward pass for efficiency. The
  // input is therefore stored as a Tensor of shape (batchSize x inputs)
  // where each row is one sample. The same weight matrix is applied to
  // every row, and the same bias is added to every row, but the matrix
  // multiply kernel from tutorial two handles the entire batch in a
  // single call. In the example below, the batch size is just 1 and the
  // input is the row vector [0, 1, ..., 9] of shape (1 x 10). Due to the nature
  // of matrix multiplication, each row in the output Tensor only depends
  // on the values within that row of the input. This means that the batch
  // dimension does not effect the output values of the network, with each row
  // remaining seperate.

  mattTorch::Tensor input({1, 10});

  for (int i{0}; i < input.getDimensions()[1]; i++) {
    input[{0, i}] = i;
  }

  std::cout << "input: \n" << input << "\n";

  std::println("-----------------------------------------------------------");

  // The FullyConnectedLayer constructor begins by constructing its
  // weight Tensor with dimensions (inputs x outputs) and its bias Tensor
  // with dimensions (outputs). Both are leaf Tensors constructed via the
  // normal Tensor constructor described in tutorial one, so both allocate
  // their own 64-byte aligned TensorStorage, and both are allocated  a
  // GradAccumulator pointing at a  gradient Tensor in exactly the way described
  // in tutorial three. After the forward pass, they become part of the
  // computational graph as learnable parameters, with their gradient slots
  // ready to receive contributions from any future backward pass.
  //
  // Their values are then filled by sampling from a normal distribution
  // whose standard deviation depends on the requested initialisation
  // strategy. Xavier initialisation uses
  //
  //     sigma = sqrt(2 / (inputs + outputs))
  //
  // which keeps the variance of the activations and the variance of the
  // gradients on roughly the same scale across layers of differing
  // fan-in and fan-out, and is well suited to layers followed by tanh
  // or other saturating activations. He initialisation uses
  //
  //     sigma = sqrt(2 / inputs)
  //
  // which compensates for the fact that ReLU zeroes out roughly half of
  // its inputs by inflating the variance by a factor of two, keeping
  // the variance of the activations approximately constant through deep
  // ReLU networks. Picking the wrong scale here is not a stylistic
  // choice: too small and the signal collapses to zero a few layers in,
  // making the gradients vanish and the network untrainable; too large
  // and the activations saturate or the gradients blow up. The
  // initialisation parameter of the FullyConnectedLayer constructor
  // takes either "xavier" or "he" as a string, defaults to "xavier",
  // and throws std::invalid_argument for anything else.
  //
  // A std::default_random_engine is seeded from the system clock so
  // successive constructions produce independent parameter draws. Each
  // parameter element is then written directly into the contiguous
  // storage via the getData pointer rather than through the bracketed
  // indexing accessors, since none of the strides arithmetic from
  // tutorial one is needed for a simple linear walk over every element.
  // The same generator is used for both the weight and the bias, so the
  // bias draws follow on directly from the weight draws in the same
  // random stream.

  mattTorch::FullyConnectedLayer fullyConnected(10, 20);

  std::cout << "weights: \n"
            << *(fullyConnected.getParameters()[0]) << std::endl;

  std::cout << "bias: \n" << *(fullyConnected.getParameters()[1]) << std::endl;

  std::println("-----------------------------------------------------------");

  // The forward pass itself is three Tensor operations. The matrix
  // multiply takes the input of shape (batchSize x inputs) and the weight
  // of shape (inputs x outputs) and produces a weighted output of shape
  // (batchSize x outputs), using the matrixMultiply kernel described in
  // tutorial two. The bias Tensor has shape (outputs), one entry per
  // output feature, and needs to be added elementwise to every row of
  // the weighted output. Rather than write a separate row-wise add
  // kernel, the bias is reshaped to match by calling broadcast(0,
  // batchSize), which inserts a new first dimension of size batchSize
  // and replicates the (outputs) row down it to produce a
  // (batchSize x outputs) Tensor that shares the bias's values across
  // the batch. This is exactly the broadcast operation from tutorial
  // two doing its intended job. The elementwise operator+ then combines
  // the weighted output and the broadcast bias into the final result.
  //
  // Each of the three operations inside forward allocates its own
  // GradFunction on its result in exactly the way described in tutorial
  // three. The matrixMultiply call allocates a GradMatrixMultiply node
  // whose savedTensors contain the input and the weight, and whose
  // nextFunctions point at the input's gradFunction and the weight's
  // GradAccumulator. The broadcast call installs a GradBroadcast node
  // whose savedTensors are the bias and whose nextFunctions point at the
  // bias's GradAccumulator. The operator+ call allocates a GradAdd node
  // whose savedTensors are the weighted output and the broadcast bias
  // and whose nextFunctions point at the two GradFunctions just
  // allocated by matrixMultiply and broadcast. A single call to
  // FullyConnectedLayer::forward therefore appends a three-node
  // subgraph to the computational graph, with the weight and bias
  // GradAccumulators sitting at the leaves of that subgraph alongside
  // the input's own GradFunction. When the eventual loss has backward
  // called on it, the chain rule traversal described in tutorial three
  // walks back through GradAdd, into GradMatrixMultiply and GradBroadcast,
  // and finally accumulates the gradients into weight.gradient and
  // bias.gradient ready for an optimiser step.

  auto output = fullyConnected.forward(input);

  std::cout << "output: \n" << output << "\n";

  std::println("-----------------------------------------------------------");

  // Not every layer has learnable parameters. The ReLU, Tanh and
  // Softmax classes are all Layers in the same sense as
  // FullyConnectedLayer but they exist purely to give a nonlinearity
  // to the forward computation. Their getNumParameters returns zero
  // and their getParameters returns an empty vector, which means the
  // optimiser introduced in a later tutorial simply has nothing to
  // update for these layers. ReLU::forward simply calls Tensor::ReLU,
  // Tanh::forward simply calls Tensor::tanh. The GradFunction allocated by the
  // underlying Tensor operation is what carries the gradient of the activation
  // back through the graph during a backward pass, so nothing extra is required
  // of the Layer itself in order for it to be differentiable.

  mattTorch::ReLU relu;
  auto reluOutput = relu.forward(output);
  std::cout << "ReLU(output): \n" << reluOutput << "\n";

  std::println("-----------------------------------------------------------");

  mattTorch::Tanh tanh;
  auto tanhOutput = tanh.forward(output);
  std::cout << "tanh(output): \n" << tanhOutput << "\n";

  std::println("-----------------------------------------------------------");

  // The forward method of Softmax is also itself a composition of Tensor
  // operations rather than a single delegated call. It first applies the
  // elementwise exponential to the input, then reduces along dimension one with
  // reductionSum to produce a per-row normalising constant, then broadcasts
  // that constant back along dimension one so it has the same shape as the
  // exponentials, and finally divides the exponentials by the broadcast sums in
  // place with operator/=. The result is a Tensor of the same shape as the
  // input in which each row sums to one, providing a probability
  // distribution across the second dimension. Each of these four
  // operations  allocates its own GradFunction in exactly the way
  // described in tutorial three, so a single Softmax forward call
  // appends a four-node subgraph to the computational graph in the same
  // way that FullyConnectedLayer::forward appends a three-node
  // subgraph. This is the pattern that the entire library uses to
  // combine single differentiable operations into higher-level
  // differentiable building blocks: the Layer class is the unit of
  // composition at the level of the network, the Tensor operation is
  // the unit of composition at the level of autograd, and the two
  // compose cleanly because every Tensor operation a Layer can call can already
  // fit into the graph.

  mattTorch::Softmax softmax;
  auto softmaxOutput = softmax.forward(reluOutput);
  std::cout << "softmax(ReLU(output)): \n" << softmaxOutput << "\n";

  std::println("-----------------------------------------------------------");

  // The getNumParameters method on each of these layers is what allows
  // a Network containing a mix of them to report its total parameter
  // count without having to know anything about the individual layer
  // types. Each layer returns the size of its own parameter set,
  // FullyConnectedLayer returning the sum of the number of elements in
  // its weight and its bias Tensors, and the activation layers each
  // returning zero. Summing these values across the layers of a
  // Network gives the total number of learnable parameters in the
  // network, which is the natural unit in which to describe the
  // capacity of a model.

  std::cout << "fully connected parameters: "
            << fullyConnected.getNumParameters() << "\n";
  std::cout << "ReLU parameters: " << relu.getNumParameters() << "\n";
  std::cout << "Tanh parameters: " << tanh.getNumParameters() << "\n";
  std::cout << "Softmax parameters: " << softmax.getNumParameters() << "\n";

  std::println("-----------------------------------------------------------");

  return 0;
}
