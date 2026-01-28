#pragma once

#include "mattTorch/tensor/tensorView/tensorView.h"
namespace mattTorch {

class Criterion {
 public:
  virtual TensorView calculateLoss(TensorView& input, TensorView& target) = 0;
  virtual ~Criterion() = default;
};

}  // namespace mattTorch
