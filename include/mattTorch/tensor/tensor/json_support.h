#pragma once
#include <mattTorch/tensor/tensor/tensor.h>

#include <nlohmann/json.hpp>
#include <plotlypp/traits.hpp>

namespace plotlypp {
template <>
struct is_plotly_data_array_extension<mattTorch::Tensor> : std::true_type {};

template <>
struct range_element_type<mattTorch::Tensor> {
  using type = double;
};
}  // namespace plotlypp

namespace mattTorch {
void to_json(nlohmann::json& j, const Tensor& t);
}  // namespace mattTorch
