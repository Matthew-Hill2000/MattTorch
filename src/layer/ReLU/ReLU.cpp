#include <immintrin.h>
#include <mattTorch/layer/ReLU/ReLU.h>

namespace mattTorch {

Tensor ReLU::forward(Tensor& inputTensor) { return inputTensor.ReLU(); }

int ReLU::getNumParameters() { return 0; }

std::vector<std::shared_ptr<Tensor>> ReLU::getParameters() { return {}; }

}  // namespace mattTorch
