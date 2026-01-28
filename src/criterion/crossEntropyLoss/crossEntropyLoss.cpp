
#include <mattTorch/criterion/crossEntropyLoss/crossEntropyLoss.h>

namespace mattTorch::criterion {

TensorView CrossEntropyLoss::calculateLoss(TensorView& input,
                                           TensorView& target) {
  TensorView logInputs = input.log();
  TensorView loss = logInputs * target;
  loss = -1 * loss;
  loss = loss.reductionSum(1);
  return loss;
}
}  // namespace mattTorch::criterion
