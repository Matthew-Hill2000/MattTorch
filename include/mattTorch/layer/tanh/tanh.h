#pragma once

#include <mattTorch/layer/layer.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch {

class Tanh : public Layer {
 public:
  Tanh() = default;
  TensorView forward(TensorView& input) override;
  std::vector<std::shared_ptr<TensorView>> getParameters() override;
  int getNumParameters() override;
};

}  // namespace mattTorch
