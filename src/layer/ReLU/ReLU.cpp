#include <immintrin.h>
#include <mattTorch/layer/ReLU/ReLU.h>

namespace mattTorch {
ReLU::ReLU() = default;

TensorView ReLU::forward(TensorView& inputTensor) {
  // Store the input using deep_copy to ensure it's preserved for backprop
  this->inputTensor = inputTensor;
  this->inputShape = inputTensor.getDimensions();

  this->outputTensor = TensorView(inputShape);

  outputTensor = inputTensor.ReLU();

  return outputTensor;
}

int ReLU::getNumParameters() { return 0; }

std::vector<std::shared_ptr<TensorView>> ReLU::getParameters() { return {}; }

}  // namespace mattTorch
