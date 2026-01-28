#pragma once
#include <cstddef>

namespace mattTorch::tensor {
class TensorStorage {
 private:
  // Stores the data associated with a TensorView
  // object in contiguous memory
  double* mData;
  size_t size;

 public:
  // Create a TensorStorage object with the capacity for "size" elements and
  // initialise all with the value given by initValue
  TensorStorage(size_t size);
  // Create a TensorStorage object by using move semantics to assume ownership
  // of a premade vector of double values
  TensorStorage(double* rValues);
  // We dont want to be able to copy objects of TensorStorage since the
  // TensorView class is the primary interface by which we interact with tensor
  // objects. The entire point of the TensorView-TensorStorage structure is to
  // share data and only have a single TensorStorage object shared.
  TensorStorage(const TensorStorage& rOther) = delete;

  ~TensorStorage();
  // return a reference to the data stored at indexed position within the data
  // vector
  double& at(int index);
  const double& at(int index) const;

  size_t getSize() const;

  // return the data stored in this tensor
  double* getData();
  const double* getData() const;
  void setAllValues(double value);
};
}  // namespace mattTorch::tensor
