#include "tensorStorage.h"

TensorStorage::TensorStorage(size_t size, double initValue)
    : mData(size, initValue) {}

TensorStorage::TensorStorage(std::vector<double>&& rValues)
    : mData(std::move(rValues)) {}

double& TensorStorage::at(int index) {
  if (index < 0 || index >= static_cast<int>(mData.size())) {
    throw std::out_of_range("Storage index out of bounds");
  }
  return mData[index];
}

const double& TensorStorage::at(int index) const {
  if (index < 0 || index >= static_cast<int>(mData.size())) {
    throw std::out_of_range("Storage index out of bounds");
  }
  return mData[index];
}

size_t TensorStorage::size() const { return mData.size(); }

std::vector<double>& TensorStorage::getData() { return mData; }

const std::vector<double>& TensorStorage::getData() const { return mData; }
