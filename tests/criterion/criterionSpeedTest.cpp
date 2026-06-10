#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"
#include "mattTorch/tensor/tensor/tensor.h"

// ============ Tests ============

void benchMSELoss(mattTorch::Tensor& input, mattTorch::Tensor& target) {
  mattTorch::criterion::MSELoss mse;
  mattTorch::Tensor loss = mse.calculateLoss(input, target);
  loss.backward();
}

void benchCrossEntropyLoss(mattTorch::Tensor& input,
                           mattTorch::Tensor& target) {
  mattTorch::criterion::CrossEntropyLoss ce;
  mattTorch::Tensor loss = ce.calculateLoss(input, target);
  loss.backward();
}

int main() {
  // == Mean Squared Error: Forwards and Backwards Speed Test ==
  const int batch = 256, features = 1024;

  mattTorch::Tensor mseInput = randomTensor({batch, features});
  mattTorch::Tensor mseTarget = randomTensor({batch, features}, false);
  run("MSELoss (fwd+bwd)", [&] { benchMSELoss(mseInput, mseTarget); });

  // == Cross Entropy Loss: Forwards and Backwards Speed Test ==
  mattTorch::Tensor ceInput = randomTensor({batch, features});

  // Need Possitive values
  for (int i = 0; i < ceInput.getNValues(); i++) {
    ceInput.setValueDirect(i, 0.1);
  }

  mattTorch::Tensor ceTarget({batch, features});
  ceTarget.setRequiresGrad(false);

  // One hot encoded
  for (int b = 0; b < batch; b++) {
    ceTarget[{b, 0}] = 1.0;
  }

  run("CrossEntropyLoss (fwd+bwd)",
      [&] { benchCrossEntropyLoss(ceInput, ceTarget); });
}
