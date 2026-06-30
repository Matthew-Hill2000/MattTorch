#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensor/tensor.h>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>

#include <cstring>
#include <nlohmann/json.hpp>

namespace mattTorch {

void to_json(nlohmann::json& j, const Tensor& t) {
  const auto n = static_cast<size_t>(t.getNValues());
  const auto* p = t.getData();
  j = std::vector<double>(p, p + n);
}
}  // namespace mattTorch
