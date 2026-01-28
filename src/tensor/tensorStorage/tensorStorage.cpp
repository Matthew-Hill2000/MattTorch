#include <cstring>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>
#include <mm_malloc.h>

#include <stdexcept>

namespace mattTorch::tensor {
// Create a TensorStorage object with the capacity for "size" elements and
// initialise all with the value given by initValue. Ensure allocated memory
// is alligned into an address that is a multiple of 32.
TensorStorage::TensorStorage(size_t size ) : size(size) {
  posix_memalign(reinterpret_cast<void**>(&mData), 32, size * sizeof(double));
  memset(mData, 0, size * sizeof(double));
}

TensorStorage::~TensorStorage() { free(this->mData); }

// Create a TensorStorage object by using move semantics to assume ownership
// of a premade vector of double values
TensorStorage::TensorStorage(double* rValues) : mData(std::move(rValues)) {}

double& TensorStorage::at(int index) {
  if (index < 0 || index >= static_cast<int>(size)) {
    throw std::out_of_range("Storage index out of bounds");
  }
  return mData[index];
}

// return a reference to the data stored at indexed position within the data
// vector
const double& TensorStorage::at(int index) const {
  if (index < 0 || index >= static_cast<int>(size)) {
    throw std::out_of_range("Storage index out of bounds");
  }
  return mData[index];
}

// Return the size of the storage for this tensor, i.e the number of elements
size_t TensorStorage::getSize() const { return size; }

// return the data stored in this tensor
double* TensorStorage::getData() { return mData; }
const double* TensorStorage::getData() const { return mData; }

void TensorStorage::setAllValues(double value) {
  std::fill(mData, mData+size, value);
}
}  // namespace mattTorch::tensor
