#pragma once

#include <mattTorch/layer/layer.h>

namespace mattTorch {

class ReLU : public Layer {
 private:
  TensorView inputTensor;
  std::vector<int> inputShape;
  TensorView outputTensor;

 public:
  ReLU();
  TensorView forward(TensorView& inputTensor) override;
  std::vector<std::shared_ptr<TensorView>> getParameters() override;
  int getNumParameters() override;
};

}  // namespace mattTorch
