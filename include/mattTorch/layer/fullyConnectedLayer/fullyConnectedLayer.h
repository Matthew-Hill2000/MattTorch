#pragma once

#include <mattTorch/layer/layer.h>

namespace mattTorch {

class FullyConnectedLayer : public Layer {
 private:
  TensorView weight;
  TensorView bias;

 public:
  FullyConnectedLayer(int inputs, int outputs);
  TensorView forward(TensorView& inputTensor);
  int getNumParameters();
  std::vector<std::shared_ptr<TensorView>> getParameters();
};

}  // namespace mattTorch
