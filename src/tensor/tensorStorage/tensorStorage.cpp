#include <mattTorch/tensor/tensorStorage/tensorStorage.h>
#include <mm_malloc.h>

#include <cstring>
#include <stdexcept>
#include <utility>

namespace mattTorch::tensor {

TensorStorage::TensorStorage(size_t size) : size{size} {
  posix_memalign(reinterpret_cast<void**>(&mData), 64, size * sizeof(double));
  memset(mData, 0, size * sizeof(double));
}

TensorStorage::~TensorStorage() { free(this->mData); }

TensorStorage::TensorStorage(double* rValues) : mData(rValues) {}

double& TensorStorage::at(int index) {
  if (index < 0 || std::cmp_greater_equal(index, size)) {
    throw std::out_of_range("Storage index out of bounds");
  }
  return mData[index];
}

const double& TensorStorage::at(int index) const {
  if (index < 0 || std::cmp_greater_equal(index, size)) {
    throw std::out_of_range("Storage index out of bounds");
  }
  return mData[index];
}

size_t TensorStorage::getSize() const { return size; }

double* TensorStorage::getData() { return mData; }

const double* TensorStorage::getData() const { return mData; }

void TensorStorage::setAllValues(double value) {
  std::fill(mData, mData + size, value);
}
}  // namespace mattTorch::tensor
