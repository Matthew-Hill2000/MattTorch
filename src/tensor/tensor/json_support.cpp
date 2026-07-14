#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensor/tensor.h>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>

#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

namespace mattTorch {

nlohmann::json tensorToJson(const double* data, const Dims& dims,
                            const Strides& strides, int offset, size_t dim) {
  nlohmann::json arr = nlohmann::json::array();
  const int extent = dims[dim];
  const int stride = strides[dim];

  if (dim + 1 == dims.size()) {
    for (int i = 0; i < extent; i++) {
      arr.push_back(data[offset + i * stride]);
    }
  } else {
    for (int i = 0; i < extent; i++) {
      arr.push_back(
          tensorToJson(data, dims, strides, offset + i * stride, dim + 1));
    }
  }
  return arr;
}

void to_json(nlohmann::json& j, const Tensor& t) {
  const Dims dims = t.getDimensions();
  if (dims.size() == 0) {
    j = nlohmann::json::array();
    return;
  }
  j = tensorToJson(t.getData(), dims, t.getStrides(), t.getOffset(), 0);
}

void saveTensorToJson(const Tensor& t, const std::string& path, int indent) {
  nlohmann::json j;
  to_json(j, t);
  std::ofstream out(path);
  out << j.dump(indent);
}
}  // namespace mattTorch
