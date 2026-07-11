#include <mattTorch/criterion/crossEntropyLoss/softmaxCrossEntropyLoss.h>

namespace mattTorch::criterion {

Tensor SoftmaxCrossEntropyLoss::calculateLoss(Tensor& input, Tensor& target) {
  double max = input.max();
  Tensor output = (input - max) - (input - max)
                                      .exponential()
                                      .reductionSum(1)
                                      .log()
                                      .broadcast(1, input.getDimensions()[1]);
  Tensor loss = (output * target).reductionSum(1).mean() * -1.0;
  return loss;
}
}  // namespace mattTorch::criterion
