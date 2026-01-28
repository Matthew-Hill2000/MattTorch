#pragma once

#include <mattTorch/layer/layer.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch {
class NetworkBuilder;


class Network {
 private:
  std::vector<Layer*> layers;  // Collection of network layers
  std::vector<std::shared_ptr<TensorView>> parameters;
  friend class NetworkBuilder;  // Builder has access to private members

 public:
  Network() = default;

  Network(std::vector<Layer*> layers);

  // Add a single layer to the network
  void addLayer(Layer* layer);

  void addParameter(std::shared_ptr<TensorView> parameter);
  std::vector<std::shared_ptr<TensorView>> getParameters();

  // Get all layers in the network
  std::vector<Layer*> getLayers() const;

  // Perform forward propagation through all layers sequentially 
  TensorView forward(TensorView& input);

  // Get summary of the network 
  std::string summary() const;

  // Save the network parameters to a csv
  void saveParameters(const std::string& path);
  // Load the network parameters from a csv
  void loadParameters(const std::string& path);
};

class NetworkBuilder {
 private:
  std::vector<Layer*> layers;  // Temporary storage for layers
  std::vector<std::shared_ptr<TensorView>> parameters;

 public:
  NetworkBuilder() = default;

  ~NetworkBuilder();

  NetworkBuilder& addReluLayer();
  NetworkBuilder& addFullyConnectedLayer(int input_size, int output_size);
  NetworkBuilder& addTanhLayer();
  NetworkBuilder& addSoftmaxLayer();
  NetworkBuilder& addLayer(Layer* layer);

  // Build and return the configured network
  Network build();
};

}  // namespace mattTorch
