#include <mattTorch/tensor/tensorStorage/tensorStorage.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <stdexcept>

#include "mattTorch/function/accumulator/gradAccumulator.h"

namespace mattTorch {

// Creates a new TensorView object indexed via the first dimension that shares
// the storage with the original tensor. Values of this tensor are
// differentiatied from the storage via the calculation of an offset that
// defines were the values associated with this subtensor begin in memory
TensorView TensorView::operator[](int index) {
  if (index < 0 || index >= dimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }

  // If the rank=1 then we create a new tensor with the single indexed value,
  // also with rank 1
  if (rank == 1) {
    return TensorView(storage, {1}, {1}, offset + index * strides[0], 1, 1,
                      gradient, gradFunction, isLeaf, requiresGrad, hasGrad);
  } else {
    // The dimensions and strides of the new tensor are the same as the old but
    // with the first elements of each of these parameters removed. The new
    // offset into the TensorStorage and the new NValues and Rank have to be
    // calculated with the first dimension removed
    std::vector<int> newDims(dimensions.begin() + 1, dimensions.end());
    std::vector<int> newStrides(strides.begin() + 1, strides.end());
    int newOffset(offset + index * strides[0]);
    int newNValues{nValues / dimensions[0]};
    int newRank{rank - 1};

    return TensorView(storage, newDims, newStrides, newOffset, newNValues,
                      newRank, gradient, gradFunction, isLeaf, requiresGrad,
                      hasGrad);
  }
}

// Creates a new TensorView object indexed via the first dimension that shares
// the storage with the original tensor. Values of this tensor are
// differentiatied from the storage via the calculation of an offset that
// defines were the values associated with this subtensor begin in memory
const TensorView TensorView::operator[](int index) const {
  if (index < 0 || index >= dimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }

  // If the rank=1 then we create a new tensor with the single indexed value,
  // also with rank 1
  if (rank == 1) {
    return TensorView(storage, {1}, {1}, offset + index * strides[0], 1, 1,
                      gradient, gradFunction, isLeaf, requiresGrad, hasGrad);
  } else {
    // The dimensions and strides of the new tensor are the same as the old but
    // with the first elements of each of these parameters removed. The new
    // offset into the TensorStorage and the new NValues and Rank have to be
    // calculated with the first dimension removed
    std::vector<int> newDims(dimensions.begin() + 1, dimensions.end());
    std::vector<int> newStrides(strides.begin() + 1, strides.end());
    int newOffset{offset + index * strides[0]};
    int newNValues{nValues / dimensions[0]};
    int newRank{rank - 1};

    return TensorView(storage, newDims, newStrides, newOffset, newNValues,
                      newRank, gradient, gradFunction, isLeaf, requiresGrad,
                      hasGrad);
  }
}

// Returns the value of a specific element within the tensor at the specified
// indices
double& TensorView::operator[](const std::vector<int>& indices) {
  // calculate the linear index and return the value in TensorStorage at said
  // linear index
  int index = calculateIndex(indices);
  return storage->at(index);
}

// Returns the value of a specific element within the tensor at the specified
// indices
const double& TensorView::operator[](const std::vector<int>& indices) const {
  // calculate the linear index and return the value in TensorStorage at said
  // linear index
  int index = calculateIndex(indices);
  return storage->at(index);
}

// Converts a vector of indices into the corresponding index of the
// appropriate element within the linear storage
int TensorView::calculateIndex(const std::vector<int>& indices) const {
  if (static_cast<int>(indices.size()) != rank) {
    throw std::invalid_argument("Number of indices doesn't match tensor rank");
  }

  for (int i{0}; i < static_cast<int>(indices.size()); i++) {
    if (indices[i] < 0 || indices[i] >= dimensions[i]) {
      throw std::out_of_range("Index out of bounds for dimension " +
                              std::to_string(i));
    }
  }

  int index{offset};
  for (int i{0}; i < rank; i++) {
    index += indices[i] * strides[i];
  }

  return index;
}

// Set the value of an element of this tensor at the specified indices
void TensorView::setValue(const std::vector<int>& indices, double value) {
  int index = calculateIndex(indices);
  storage->at(index) = value;
}

// Get the value of an element of this tensor at the specified indices
double TensorView::getValue(const std::vector<int>& indices) const {
  int index = calculateIndex(indices);
  return storage->at(index);
}

// Directly set the value of an element in this tensor by specifying the
// linear index value directly
void TensorView::setValueDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  storage->at(offset + linearIndex) = value;
}

// Directly get the value of an element in this tensor by specifying the
// linear index value directly
double TensorView::getValueDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return storage->at(offset + linearIndex);
}

// Directly set the value of an element of this tensors gradient by specifying
// the linear index value directly
void TensorView::setGradientDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  gradient->setValueDirect(offset + linearIndex, value);
}

// Directly get the value of an element of this tensors gradient by specifying
// the linear index value directly
double TensorView::getGradientDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return gradient->getValueDirect(offset + linearIndex);
}

// Getters and setters for the class attributes

double* TensorView::getData() const { return this->storage->getData(); }
double* TensorView::getGradientData() const {
  return this->gradient->getData();
}

bool TensorView::getHasGrad() { return this->hasGrad; }

void TensorView::setHasGrad(bool hasGrad) { this->hasGrad = hasGrad; }

std::vector<int> TensorView::getStrides() const { return strides; }

int TensorView::getOffset() const { return offset; }

int TensorView::getNValues() const { return nValues; }

int TensorView::getRank() const { return rank; }

std::vector<int> TensorView::getDimensions() const { return dimensions; }

void TensorView::setLeaf(bool isLeaf) { this->isLeaf = isLeaf; }

void TensorView::setRequiresGrad(bool requiresGrad) {
  this->requiresGrad = requiresGrad;
}

bool TensorView::getRequiresGrad() const { return requiresGrad; }

void TensorView::setGradFunction(std::shared_ptr<GradFunction> gradFunction) {
  this->gradFunction = std::move(gradFunction);
}

TensorView TensorView::detachGradient() {
  if (gradient == nullptr) {
    throw std::invalid_argument("gradient is nullptr");
  }

  TensorView oldGradient = *gradient;

  gradient = std::make_shared<TensorView>(dimensions, false);

  std::static_pointer_cast<function::GradAccumulator>(gradFunction)
      ->setGradient(gradient);

  return oldGradient;
}

std::shared_ptr<GradFunction> TensorView::getGradFunction() {
  return this->gradFunction;
}
}  // namespace mattTorch
