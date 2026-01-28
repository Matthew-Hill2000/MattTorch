
#include <immintrin.h>
#include <mattTorch/layer/softmax/softmax.h>

namespace mattTorch {
Softmax::Softmax() = default;

TensorView Softmax::forward(TensorView& inputTensor) {
  this->inputTensor = inputTensor;
  this->inputShape = inputTensor.getDimensions();

  this->outputTensor = TensorView(inputShape);

 
  outputTensor = inputTensor.exponential();
  TensorView summedExponentials = outputTensor.reductionSum(1);
  summedExponentials = summedExponentials.broadcast(1, outputTensor.getDimensions()[1]);
  outputTensor /= summedExponentials;
  

  return outputTensor;
}

int Softmax::getNumParameters() { return 0; }

std::vector<std::shared_ptr<TensorView>> Softmax::getParameters() { return {}; }

}  // namespace mattTorch
