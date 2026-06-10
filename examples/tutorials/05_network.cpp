#include <mattTorch/mattTorch.h>

#include <iostream>
#include <print>

#include "mattTorch/layer/ReLU/ReLU.h"
#include "mattTorch/layer/fullyConnectedLayer/fullyConnectedLayer.h"
#include "mattTorch/layer/tanh/tanh.h"
#include "mattTorch/network/network.h"
#include "mattTorch/tensor/tensor/tensor.h"

int main(int argc, char* argv[]) {
  // The previous tutorial established that an individual Layer takes a
  // Tensor input and produces a Tensor output, appending a small
  // subgraph to the computational graph along the way. A neural network
  // is fundamentally a sequence of such layers applied one after the
  // other, where the output Tensor of one layer becomes the input Tensor
  // of the next. The Network class is the container that holds an
  // ordered collection of layers, applues the forward pass through them
  // in sequence, and organises all their learnable parameters into a single
  // list that the optimiser in the next tutorial will utilise.
  //
  // A Network holds two attributes. The first is a std::vector<Layer*> of raw
  // pointers to the layers of the network in the order they are to be applied.
  // The second is a std::vector<std::shared_ptr<Tensor>> of pointers to every
  // learnable parameter Tensor that any of those layers returns through its
  // getParameters method. The Network is the owner of its Layer
  // objects, it takes them from the NetworkBuilder described below and
  // is responsible for deleting them when it is done with them, so a
  // raw pointer is sufficient. The parameter Tensors on the other hand
  // are owned jointly with the layers themselves and with the optimiser
  // introduced in the next tutorial, so a shared pointer is needed to
  // keep the underlying TensorStorage alive as long as any of those
  // parties still holds a reference to it. As discussed in tutorial
  // four, this is also what allows an update applied to a parameter
  // Tensor by the optimiser to be immediately visible to the layer
  // on the next forward pass, since both ends of the
  // arrangement point at the same TensorStorage.
  //
  // The forward method of Network iterates through its layer vector in order.
  // The empty-network case is handled first by returning the input Tensor
  // unchanged. Otherwise the first layer is called with the input and its
  // returned Tensor is saved to the Tensor current, and a loop then
  // reassigns that variable to the result of calling forward on each remaining
  // layer with the previous result. The polymorphism described in tutorial four
  // does its job here: each layers[i]->forward(...) call goes through the
  // vtable of whatever Layer subclass is actually at that index, so a single
  // loop handles a sequence of FullyConnectedLayer, ReLU, Tanh and Softmax
  // instances without any conditional logic.
  //
  // Crucially, the autograd graph is constructed automatically in
  // this loop. Each layer's forward adds its own GradFunction nodes
  // to its result Tensor as described in tutorial four, and because
  // the next layer's forward uses that Tensor as its input, the
  // new GradFunction nodes appended by the next layer have their
  // nextFunctions pointing at the GradFunctions of the previous layer. By the
  // time the loop finishes, the final returned Tensor is at the end of a single
  // combined graph whose leaves are every FullyConnectedLayer's weight and bias
  // GradAccumulators, and the single backward pass invoked on whatever loss
  // this output feeds into will accumulate gradients into every one of those
  // leaves in one sweep, exactly as described in tutorial three for an
  // arbitrary graph.

  std::println("-----------------------------------------------------------");

  // Network objects are intended to be constructed via the
  // NetworkBuilder class, which provides an  interface for assembling layers
  // and collecting their parameters. The builder is itself an object
  // holding the same two vectors that the Network does, std::vector<Layer*> for
  // layers and std::vector<std::shared_ptr<Tensor>> for parameters. Each
  // addXLayer method on the builder constructs a new layer of the
  // requested type on the heap with operator new, adds the raw
  // pointer to the layers vector, calls the new layer's getParameters
  // and appends each of the returned shared pointers into the
  // parameters vector, and finally returns a reference to the builder
  // itself. Returning the builder itself by reference is what enables the
  // build chaining in which an entire network can be specified as a
  // single expression of chained addX().addY().build() methods.
  //
  // The activation layers contain no learnable parameters, so their
  // getParameters returns an empty vector and the builder's parameter
  // list is unchanged by their addition. Only the FullyConnectedLayer
  // adds to the parameter list, pushing its weight and its bias
  // shared pointers in that order. The addLayer overload that accepts
  // an arbitrary Layer*  allows for easy addition of user-defined
  // layers: the builder takes the raw pointer, appends it to the
  // layers vector exactly as the add methods do, and adds whatever parameter
  // Tensors the custom layer contains through its getParameters into the
  // parameter vector. A null pointer is rejected with a std::invalid_argument.
  //
  // The builder owns its layers from the moment they are added until
  // either build is called or the builder itself is destroyed. The
  // destructor of NetworkBuilder walks the layers vector and calls
  // delete on every entry that is still present at the time of
  // destruction. This means that a builder that goes out of scope
  // without ever being built does not have a memory leak of its layers, but it
  // also means that any layers that were transferred to a Network via build are
  // no longer in the builder's vector at destruction time and are therefore not
  // deleted twice. The transfer is performed by build by iterating through the
  // builder's vectors and pushing every layer pointer into the Network via
  // addLayer and every parameter shared pointer via addParameter, and then
  // clearing the builder's layers vector so that the destructor finds nothing
  // left to delete. The build method returns the resulting Network by value,
  // taking advantage of the same NRVO and move semantics described in tutorial
  // two to avoid copying the underlying vectors.

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(4, 8)
                               .addReluLayer()
                               .addFullyConnectedLayer(8, 8)
                               .addTanhLayer()
                               .addFullyConnectedLayer(8, 4)
                               .build();

  // The expression above constructs a temporary NetworkBuilder, chains
  // six addX calls into it (each returning the same NetworkBuilder& so
  // the next call can be attached), and finishes with a build that
  // transfers ownership of the layers and parameters into the returned Network.
  // The temporary NetworkBuilder then immediately goes out of scope, but its
  // layers vector was cleared by build and its destructor has nothing left to
  // delete. The resulting Network is the named object net, holding five Layer*
  // entries (three FullyConnectedLayer, one ReLU, one Tanh) and six
  // parameter shared pointers (the weight and bias of each
  // FullyConnectedLayer in the order they were added).

  std::println("-----------------------------------------------------------");

  // The summary method generates a human readable description of the
  // network. It iterates through layers vector and, for each entry, uses
  // dynamic_cast to test the pointer against each of the known concrete layer
  // types in turn. dynamic_cast is the C++ mechanism for safe runtime type
  // identification of polymorphic types: it inspects the vtable pointer of the
  // object behind the pointer and returns a non-null result if and only if the
  // dynamic type of the object is the requested derived type or a further
  // derived class of it.
  //
  // The current implementation matches against FullyConnectedLayer, ReLU and
  // Tanh in turn and falls through to a literal "Unknown Layer Type" label for
  // anything that does not match. Custom layers added via addLayer therefore
  // appear in the summary under that generic label until an explicit case for
  // them is added to the method. The output is returned as a single std::string
  // assembled from a std::stringstream, so the caller can route it to
  // std::cout, a log file or any other sink.

  std::cout << net.summary();
  std::println("-----------------------------------------------------------");

  // A Tensor passed into Network::forward propagates through every layer in the
  // order they were added. The input here is a batch of two samples with four
  // features each. The first FullyConnectedLayer multiplies it by a (4 x 8)
  // weight, adds a broadcast (8) bias, and produces a (2 x 8) intermediate. The
  // ReLU layer applies the elementwise max(0, x). The second
  // FullyConnectedLayer takes that (2 x 8) result through a (8 x 8) weight and
  // an (8) bias to give another (2 x 8) intermediate. The Tanh layer applies
  // elementwise tanh. The final FullyConnectedLayer multiplies by a (8 x 4)
  // weight, adds a (4) bias, and produces the (2 x 4) network output. The
  // Tensor that comes back from forward carries the entire computational graph
  // for the network attached through its gradFunction, and calling backward on
  // it (or on a scalar loss derived from it) would propagate gradients into the
  // weight and bias GradAccumulators of all three FullyConnectedLayer instances
  // in a single pass.

  mattTorch::Tensor input({2, 4});
  for (int i{0}; i < input.getDimensions()[0]; i++) {
    for (int j{0}; j < input.getDimensions()[1]; j++) {
      input[{i, j}] = i + j;
    }
  }

  std::cout << "input: \n" << input << "\n";

  auto output = net.forward(input);
  std::cout << "network output: \n" << output << "\n";

  std::println("-----------------------------------------------------------");

  // You can access the Networks full parameter list through getParameters,
  // returning the std::vector<std::shared_ptr<Tensor>> that the builder
  // created at construction time. This is the vector that the optimiser will
  // be constructed from in the next tutorial. The order of the parameter
  // Tensors within this vector matches the order in which they were added
  // during construction. As the builder walked through the three
  // addFullyConnectedLayer calls in turn, the weight and bias shared pointers
  // of each layer were pushed in that order, so the resulting vector lays out
  // as
  //
  //     [W0, b0, W1, b1, W2, b2]
  //
  // for this three-fully-connected-layer network. The two activation
  // layers contribute nothing to this list. The total number of
  // learnable scalar values across the network is the sum over each
  // parameter Tensor of getNValues, which equals 4*8 + 8 + 8*8 + 8 +
  // 8*4 + 4 = 148 for the network constructed above.

  auto parameters = net.getParameters();
  std::cout << "total parameter Tensors: " << parameters.size() << "\n";
  int totalValues{0};
  for (auto& parameter : parameters) {
    totalValues += parameter->getNValues();
  }
  std::cout << "total parameter values: " << totalValues << "\n";

  std::println("-----------------------------------------------------------");

  // saveParameters and loadParameters save and load the values of every
  // parameter Tensor in the network to and from a CSV file. The file
  // format is one line per parameter Tensor, with the entries of that
  // Tensor written in the order of its contiguous data storage,
  // separated by commas. saveParameters iterates through the parameters vector
  // and for each parameter Tensor obtains the raw double* via getData
  // and writes its getNValues entries to the file in linear order. No
  // shape information is recorded in the file, only the raw values,
  // since the receiving network is required to have been built with
  // the same architecture as the saving network in order for the load
  // to match up.
  //
  // loadParameters iterates through the file line by line in tandem with the
  // parameters vector. For each line it constructs a std::stringstream,
  // splits it on commas, parses each value with std::stod, and writes the
  // result directly into the contiguous data storage of the corresponding
  // parameter Tensor via getData. This direct write bypasses the Tensor
  // arithmetic operators entirely, so no GradFunction is installed and the
  // autograd graph is unaffected. Both methods throw std::runtime_error if the
  // file cannot be opened.
  //
  // Saving the parameters of a network into a file and loading them back into a
  // freshly constructed network of the same architecture therefore reproduces
  // the exact same forward behaviour, since the weight and bias values of every
  // FullyConnectedLayer are identical between the two. The example below builds
  // an identical second network from scratch, loads the saved parameters into
  // it, and confirms that the output matches the original.

  net.saveParameters("parameters.csv");

  mattTorch::Network loaded = mattTorch::NetworkBuilder()
                                  .addFullyConnectedLayer(4, 8)
                                  .addReluLayer()
                                  .addFullyConnectedLayer(8, 8)
                                  .addTanhLayer()
                                  .addFullyConnectedLayer(8, 4)
                                  .build();

  loaded.loadParameters("parameters.csv");

  auto reloadedOutput = loaded.forward(input);
  std::cout << "reloaded network output: \n" << reloadedOutput << "\n";

  std::println("-----------------------------------------------------------");

  return 0;
}
