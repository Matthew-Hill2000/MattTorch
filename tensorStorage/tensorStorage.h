#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <iostream>
#include <memory>
#include <vector>
class TensorStorage {
 private:
  std::vector<double> mData;

 public:
  TensorStorage(size_t size, double initValue = 0.0);
  TensorStorage(std::vector<double>&& rValues);
  TensorStorage(const TensorStorage& rOther) = delete;

  double& at(int index);
  const double& at(int index) const;

  size_t size() const;

  std::vector<double>& getData();
  const std::vector<double>& getData() const;
};
#endif
