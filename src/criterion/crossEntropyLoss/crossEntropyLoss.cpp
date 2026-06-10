
#include <mattTorch/criterion/crossEntropyLoss/crossEntropyLoss.h>

namespace mattTorch::criterion {

Tensor CrossEntropyLoss::calculateLoss(Tensor& input, Tensor& target) {
  Tensor logInputs = input.log();
  Tensor loss = logInputs * target;
  loss = -1 * loss;
  loss = loss.reductionSum(1);
  return loss;
}
}  // namespace mattTorch::criterion
