#include <mattTorch/layer/tanh/tanh.h>

namespace mattTorch {

TensorView Tanh::forward(TensorView& input) { return input.tanh(); }

std::vector<std::shared_ptr<TensorView>> Tanh::getParameters() { return {}; }

int Tanh::getNumParameters() { return 0; }

}  // namespace mattTorch
