
#include <immintrin.h>
#include <mattTorch/layer/softmax/softmax.h>

namespace mattTorch {

Tensor Softmax::forward(Tensor& inputTensor) {
  Tensor outputTensor = inputTensor.exponential();
  Tensor summedExponentials = outputTensor.reductionSum(1);
  summedExponentials =
      summedExponentials.broadcast(1, outputTensor.getDimensions()[1]);
  outputTensor /= summedExponentials;

  return outputTensor;
}

int Softmax::getNumParameters() { return 0; }

std::vector<std::shared_ptr<Tensor>> Softmax::getParameters() { return {}; }

}  // namespace mattTorch
