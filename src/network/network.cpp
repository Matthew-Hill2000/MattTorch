#include <mattTorch/layer/ReLU/ReLU.h>
#include <mattTorch/layer/fullyConnectedLayer/fullyConnectedLayer.h>
#include <mattTorch/layer/layer.h>
#include <mattTorch/layer/tanh/tanh.h>
#include <mattTorch/network/network.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "mattTorch/layer/softmax/softmax.h"

namespace mattTorch {

Network::Network(std::vector<Layer*> layers) : layers{layers} {}

void Network::addLayer(Layer* layer) { layers.push_back(layer); }

void Network::addParameter(std::shared_ptr<TensorView> parameter) {
  parameters.push_back(parameter);
}

std::vector<std::shared_ptr<TensorView>> Network::getParameters() {
  return parameters;
}

std::vector<Layer*> Network::getLayers() const { return layers; }

TensorView Network::forward(TensorView& input) {
  // No layers case
  if (layers.empty()) {
    return input;
  }

  // Pass through first layer
  TensorView current = layers[0]->forward(input);

  // Pass through remaining layers
  for (int layer{1}; layer < static_cast<int>(layers.size()); layer++) {
    current = layers[layer]->forward(current);
  }

  return current;
}

NetworkBuilder::~NetworkBuilder() {
  for (auto layer : layers) {
    delete layer;
  }
  layers.clear();
}

NetworkBuilder& NetworkBuilder::addReluLayer() {
  layers.push_back(new ReLU());
  return *this;
}

NetworkBuilder& NetworkBuilder::addTanhLayer() {
  layers.push_back(new Tanh());
  return *this;
}
NetworkBuilder& NetworkBuilder::addSoftmaxLayer() {
  layers.push_back(new Softmax());
  return *this;
}

NetworkBuilder& NetworkBuilder::addFullyConnectedLayer(int inputSize,
                                                       int outputSize) {
  FullyConnectedLayer* newFCL = new FullyConnectedLayer(inputSize, outputSize);
  layers.push_back(newFCL);
  std::vector<std::shared_ptr<TensorView>> newParams = newFCL->getParameters();
  parameters.insert(parameters.end(), newParams.begin(), newParams.end());
  return *this;
}

NetworkBuilder& NetworkBuilder::addLayer(Layer* layer) {
  if (layer == nullptr) {
    throw std::invalid_argument("Cannot add nullptr as a layer");
  }
  layers.push_back(layer);
  std::vector<std::shared_ptr<TensorView>> newParams = layer->getParameters();
  parameters.insert(parameters.end(), newParams.begin(), newParams.end());
  return *this;
}

Network NetworkBuilder::build() {
  Network network;

  for (auto layer : layers) {
    network.addLayer(layer);
  }

  for (auto parameter : parameters) {
    network.addParameter(parameter);
  }

  layers.clear();

  return network;
}

std::string Network::summary() const {
  std::stringstream ss;
  ss << "Network Summary:" << std::endl;
  ss << "----------------" << std::endl;
  ss << "Total layers: " << layers.size() << std::endl;

  for (int i{0}; i < static_cast<int>(layers.size()); ++i) {
    ss << "Layer " << i + 1 << ": ";
    // Using RTTI to determine layer type
    const auto& layer = layers[i];
    if (dynamic_cast<FullyConnectedLayer*>(layer)) {
      ss << "Fully Connected Layer";
    } else if (dynamic_cast<ReLU*>(layer)) {
      ss << "ReLU Activation";
    } else if (dynamic_cast<Tanh*>(layer)) {
      ss << "Tanh Activation";
    } else {
      ss << "Unknown Layer Type";
    }
    ss << std::endl;
  }

  return ss.str();
}

void Network::saveParameters(const std::string& path) {
  std::ofstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + path);
  }

  for (auto param : this->parameters) {
    std::string params;
    double* paramData = param->getData();
    for (int i{0}; i < param->getNValues(); i++) {
      file << paramData[i];
      if (i != param->getNValues() - 1) {
        file << ",";
      }
    }
    file << "\n";
  }
}

void Network::loadParameters(const std::string& path) {
  std::ifstream file(path);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + path);
  }
  std::string line;
  int i = 0;
  while (std::getline(file, line)) {
    double* paramData = this->parameters[i]->getData();
    std::stringstream ss(line);
    std::string cell;
    int j = 0;
    while (std::getline(ss, cell, ',')) {
      double value = std::stod(cell);
      paramData[j] = value;
      j++;
    }
    i++;
  }
}
}  // namespace mattTorch
