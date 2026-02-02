#include <immintrin.h>
#include <mattTorch/layer/fullyConnectedLayer/fullyConnectedLayer.h>

#include <chrono>
#include <memory>
#include <random>

namespace mattTorch {

FullyConnectedLayer::FullyConnectedLayer(int inputs, int outputs,
                                         std::string initialisation) {
  this->weight = TensorView({inputs, outputs});
  this->bias = TensorView({outputs});

  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);

  std::normal_distribution<double> normalDistribution;
  if (initialisation == "xavier") {
    normalDistribution = std::normal_distribution<double>(
        0.0, std::sqrt(2.0 / (inputs + outputs)));
  } else if (initialisation == "he") {
    normalDistribution =
        std::normal_distribution<double>(0.0, std::sqrt(2.0 / inputs));
  } else {
    throw(std::invalid_argument(
        "initialisation for fully connected layer should be 'xavier' or 'he'"));
  }

  double* weightData = weight.getData();
  for (int i{0}; i < weight.getNValues(); i++) {
    weightData[i] = normalDistribution(generator);
  }

  double* biasData = bias.getData();
  for (int i{0}; i < bias.getNValues(); i++) {
    biasData[i] = normalDistribution(generator);
  }
}

TensorView FullyConnectedLayer::forward(TensorView& inputTensor) {
  TensorView outputTensorWeighted = inputTensor.matrixMultiply(weight);
  TensorView broadcastedBias =
      bias.broadcast(0, inputTensor.getDimensions()[0]);
  TensorView outputTensorBiased = outputTensorWeighted + broadcastedBias;
  return outputTensorBiased;
}

std::vector<std::shared_ptr<TensorView>> FullyConnectedLayer::getParameters() {
  std::vector<std::shared_ptr<TensorView>> parameters{
      std::make_shared<TensorView>(weight), std::make_shared<TensorView>(bias)};
  return parameters;
}

int FullyConnectedLayer::getNumParameters() {
  return (weight.getNValues() + bias.getNValues());
}

}  // namespace mattTorch
