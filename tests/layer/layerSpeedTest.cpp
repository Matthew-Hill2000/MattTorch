#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchReLU(mattTorch::Tensor& input) {
  mattTorch::ReLU relu;
  mattTorch::Tensor out = relu.forward(input);
  out.backward();
}

void benchTanh(mattTorch::Tensor& input) {
  mattTorch::Tanh tanhLayer;
  mattTorch::Tensor out = tanhLayer.forward(input);
  out.backward();
}

void benchSoftmax(mattTorch::Tensor& input) {
  mattTorch::Softmax softmax;
  mattTorch::Tensor out = softmax.forward(input);
  out.backward();
}

void benchFullyConnected(mattTorch::FullyConnectedLayer& fc,
                         mattTorch::Tensor& input) {
  mattTorch::Tensor out = fc.forward(input);
  out.backward();
}

int main() {
  const int batch = 1024, features = 1024;
  mattTorch::Tensor input = randomTensor({batch, features});

  run("ReLU (fwd+bwd)", [&] { benchReLU(input); });
  run("Tanh (fwd+bwd)", [&] { benchTanh(input); });
  run("Softmax (fwd+bwd)", [&] { benchSoftmax(input); });

  mattTorch::FullyConnectedLayer fc(features, features);
  run("FullyConnected (fwd+bwd)", [&] { benchFullyConnected(fc, input); });
}
