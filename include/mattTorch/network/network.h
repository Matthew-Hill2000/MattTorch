#pragma once
#include <mattTorch/layer/layer.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch {

class NetworkBuilder;

/**
 * @brief A sequential neural network composed of an ordered collection of
 * layers.
 *
 * The Network class represents a feedforward neural network in which an input
 * tensor is passed sequentially through each layer in the order they were
 * added. The Network holds raw pointers to its Layer objects as well as shared
 * pointers to all learnable parameter tensors across the network. Network
 * objects are intended to be constructed via the NetworkBuilder class, which
 * provides a fluent interface for assembling layers and collecting their
 * parameters.
 *
 * The Network class also provides functionality for saving and loading the
 * learned parameter values to and from CSV files, as well as generating a
 * text summary of the network architecture.
 *
 * @see NetworkBuilder for the builder class used to construct Network objects.
 * @see Layer for the base class from which all network layers derive.
 * @see Optimiser for the classes that optimise the parameters of this network.
 */
class Network {
 private:
  /// Raw pointers to the layers of the network in sequential order
  std::vector<Layer*> layers;

  /// Shared pointers to all learnable parameter tensors across the network
  std::vector<std::shared_ptr<TensorView>> parameters;

  /// The NetworkBuilder class is declared as a friend so that it can access
  /// the private members of Network during construction
  friend class NetworkBuilder;

 public:
  /// @brief Default constructor creating an empty Network
  Network() = default;

  /**
   * @brief Constructs a Network from a vector of layer pointers
   *
   * @param layers A vector of raw pointers to the layers of the network in
   * sequential order
   */
  Network(std::vector<Layer*> layers);

  /**
   * @brief Adds a single layer to the end of the network
   *
   * @param layer A raw pointer to the layer to add
   */
  void addLayer(Layer* layer);

  /**
   * @brief Adds a learnable parameter tensor to the network
   *
   * @param parameter A shared pointer to the TensorView representing the
   * learnable parameter to add
   */
  void addParameter(std::shared_ptr<TensorView> parameter);

  /**
   * @brief Returns all learnable parameter tensors in the network
   *
   * @return A vector of shared pointers to the TensorView objects representing
   * all learnable parameters across the network
   */
  std::vector<std::shared_ptr<TensorView>> getParameters();

  /**
   * @brief Returns all layers in the network
   *
   * @return A vector of raw pointers to the layers of the network in
   * sequential order
   */
  std::vector<Layer*> getLayers() const;

  /**
   * @brief Performs forward propagation through all layers sequentially
   *
   * Passes the input tensor through each layer of the network in the order
   * they were added, feeding the output of each layer as the input to the
   * next. If the network contains no layers, the input tensor is returned
   * unchanged.
   *
   * @param input The input tensor to pass through the network
   * @return The output tensor produced by the final layer of the network
   */
  TensorView forward(TensorView& input);

  /**
   * @brief Generates a text summary of the network architecture
   *
   * Returns a string describing the total number of layers in the network
   * and the type of each layer, determined via RTTI dynamic casting.
   *
   * @return A string containing the network summary
   */
  std::string summary() const;

  /**
   * @brief Saves all network parameter values to a CSV file
   *
   * Writes each parameter tensor as a single row of comma-separated double
   * values to the specified file path. Each row corresponds to one parameter
   * tensor, with values written in the order of the contiguous data storage.
   *
   * @param path The file path to save the parameters to
   *
   * @throws std::runtime_error If the file cannot be opened for writing
   */
  void saveParameters(const std::string& path);

  /**
   * @brief Loads network parameter values from a CSV file
   *
   * Reads each line of the specified CSV file and populates the corresponding
   * parameter tensor with the comma-separated double values. The file is
   * expected to have the same format as produced by saveParameters, with each
   * row corresponding to one parameter tensor in the same order.
   *
   * @param path The file path to load the parameters from
   *
   * @throws std::runtime_error If the file cannot be opened for reading
   */
  void loadParameters(const std::string& path);
};

/**
 * @brief A builder class for constructing Network objects via a fluent
 * interface.
 *
 * The NetworkBuilder class provides a convenient way to assemble a neural
 * network layer by layer using method chaining. Layers and their associated
 * learnable parameters are accumulated within the builder, and a final call
 * to build() constructs the Network object by transferring all layers and
 * parameters to it. The builder takes ownership of the Layer objects created
 * via its add methods and deallocates them in its destructor if build() has
 * not been called.
 *
 * @see Network for the class that is constructed by this builder.
 * @see Layer for the base class from which all network layers derive.
 */
class NetworkBuilder {
 private:
  /// Temporary storage for layer pointers during network construction
  std::vector<Layer*> layers;

  /// Temporary storage for learnable parameter tensors collected from layers
  /// during network construction
  std::vector<std::shared_ptr<TensorView>> parameters;

 public:
  /// @brief Default constructor creating an empty NetworkBuilder
  NetworkBuilder() = default;

  /**
   * @brief Destructor that deallocates any layers that have not been
   * transferred to a Network via build()
   */
  ~NetworkBuilder();

  /**
   * @brief Adds a ReLU activation layer to the network
   *
   * @return Reference to this NetworkBuilder for method chaining
   */
  NetworkBuilder& addReluLayer();

  /**
   * @brief Adds a fully connected (dense) layer to the network
   *
   * Creates a new FullyConnectedLayer with the specified input and output
   * sizes and collects its learnable parameters (weights and biases) into
   * the builder's parameter list.
   *
   * @param input_size The number of input features to the layer
   * @param output_size The number of output features from the layer
   * @return Reference to this NetworkBuilder for method chaining
   */
  NetworkBuilder& addFullyConnectedLayer(int input_size, int output_size);

  /**
   * @brief Adds a Tanh activation layer to the network
   *
   * @return Reference to this NetworkBuilder for method chaining
   */
  NetworkBuilder& addTanhLayer();

  /**
   * @brief Adds a Softmax activation layer to the network
   *
   * @return Reference to this NetworkBuilder for method chaining
   */
  NetworkBuilder& addSoftmaxLayer();

  /**
   * @brief Adds a custom layer to the network
   *
   * Adds the provided layer and collects any learnable parameters it
   * exposes into the builder's parameter list.
   *
   * @param layer A raw pointer to the layer to add
   * @return Reference to this NetworkBuilder for method chaining
   *
   * @throws std::invalid_argument If the provided layer pointer is nullptr
   */
  NetworkBuilder& addLayer(Layer* layer);

  /**
   * @brief Builds and returns the configured Network
   *
   * Transfers all accumulated layers and parameters to a new Network object
   * and clears the builder's internal state. After calling build(), the
   * builder's layer and parameter lists are empty.
   *
   * @return The constructed Network containing all added layers and parameters
   */
  Network build();
};
}  // namespace mattTorch
