#include <iostream>
#include <mattTorch/criterion/mseLoss/mseLoss.h>

namespace mattTorch::criterion {

TensorView MSELoss::calculateLoss(TensorView& input, TensorView& target) {
  TensorView difference = input - target;

  std::cout << difference << std::endl;
  TensorView differenceSquared = difference * difference;
  std::cout << differenceSquared << std::endl;
  TensorView Loss = differenceSquared.reductionSum(1);
  std::cout << Loss << std::endl;
  Loss = Loss / (input.getNValues());
  std::cout << Loss << std::endl;
  return Loss;
}
}  // namespace mattTorch::criterion
