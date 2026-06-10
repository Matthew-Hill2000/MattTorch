#include <mattTorch/criterion/mseLoss/mseLoss.h>

namespace mattTorch::criterion {

Tensor MSELoss::calculateLoss(Tensor& input, Tensor& target) {
  Tensor difference = input - target;
  Tensor differenceSquared = difference * difference;
  Tensor Loss = differenceSquared.mean();
  return Loss;
}
}  // namespace mattTorch::criterion
