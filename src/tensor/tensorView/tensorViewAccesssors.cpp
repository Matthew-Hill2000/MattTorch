#include <mattTorch/tensor/tensorStorage/tensorStorage.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <stdexcept>

#include "mattTorch/function/accumulator/gradAccumulator.h"

namespace mattTorch {

TensorView TensorView::operator[](int index) {
  if (index < 0 || index >= dimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }

  if (rank == 1) {
    return TensorView(storage, {1}, {1}, offset + index * strides[0], 1, 1,
                      gradient, gradFunction, isLeaf, requiresGrad, hasGrad);
  } else {
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

const TensorView TensorView::operator[](int index) const {
  if (index < 0 || index >= dimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }

  if (rank == 1) {
    return TensorView(storage, {1}, {1}, offset + index * strides[0], 1, 1,
                      gradient, gradFunction, isLeaf, requiresGrad, hasGrad);
  } else {
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

double& TensorView::operator[](const std::vector<int>& indices) {
  int index = calculateIndex(indices);
  return storage->at(index);
}

const double& TensorView::operator[](const std::vector<int>& indices) const {
  int index = calculateIndex(indices);
  return storage->at(index);
}

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

void TensorView::setValue(const std::vector<int>& indices, double value) {
  int index = calculateIndex(indices);
  storage->at(index) = value;
}

double TensorView::getValue(const std::vector<int>& indices) const {
  int index = calculateIndex(indices);
  return storage->at(index);
}

void TensorView::setValueDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  storage->at(offset + linearIndex) = value;
}

double TensorView::getValueDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return storage->at(offset + linearIndex);
}

void TensorView::setGradientDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  gradient->setValueDirect(offset + linearIndex, value);
}

double TensorView::getGradientDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= nValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return gradient->getValueDirect(offset + linearIndex);
}

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
