#include <mattTorch/layer/tanh/tanh.h>

namespace mattTorch {

Tensor Tanh::forward(Tensor& input) { return input.tanh(); }

std::vector<std::shared_ptr<Tensor>> Tanh::getParameters() { return {}; }

int Tanh::getNumParameters() { return 0; }

}  // namespace mattTorch
