#pragma once

#include "mattTorch/tensor/tensorView/tensorView.h"

namespace mattTorch {

class Layer {
 public:
  virtual TensorView forward(TensorView& inputTensor) = 0;
  virtual int getNumParameters() = 0;
  virtual std::vector<std::shared_ptr<TensorView>> getParameters() = 0;
  virtual ~Layer() {}
};
}  // namespace mattTorch
