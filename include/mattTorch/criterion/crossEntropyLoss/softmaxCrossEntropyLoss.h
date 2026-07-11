#pragma once
#include <mattTorch/criterion/criterion.h>
#include <mattTorch/tensor/tensor/tensor.h>
namespace mattTorch::criterion {
class SoftmaxCrossEntropyLoss : public Criterion {
 public:
  Tensor calculateLoss(Tensor& input, Tensor& target) override;
};
}  // namespace mattTorch::criterion
