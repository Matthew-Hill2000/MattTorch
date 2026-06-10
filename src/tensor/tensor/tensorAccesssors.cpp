#include <mattTorch/tensor/tensor/tensor.h>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>

#include <memory>
#include <stdexcept>

#include "mattTorch/function/accumulator/gradAccumulator.h"

namespace mattTorch {

Tensor Tensor::operator[](int index) {
  if (index < 0 || index >= tensorData.dimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }

  Dims newDims = getRank() == 1 ? Dims{1}
                                : Dims(tensorData.dimensions.begin() + 1,
                                       tensorData.dimensions.end());

  Strides newStrides = getRank() == 1 ? Strides{1}
                                      : Strides(tensorData.strides.begin() + 1,
                                                tensorData.strides.end());

  int newOffset(tensorData.offset + (index * tensorData.strides[0]));

  return {storage, TensorData{newDims, newStrides, newOffset}, gradient,
          gradFunction, gradData};
}

double& Tensor::operator[](const Dims& indices) {
  int index = calculateIndex(indices);
  return storage->at(index);
}

const double& Tensor::operator[](const Dims& indices) const {
  int index = calculateIndex(indices);
  return storage->at(index);
}

int Tensor::calculateIndex(const Dims& indices) const {
  if (indices.size() != getRank()) {
    throw std::invalid_argument("Number of indices doesn't match tensor rank");
  }

  for (std::size_t i{0}; i < indices.size(); i++) {
    if (indices[i] < 0 || indices[i] >= tensorData.dimensions[i]) {
      throw std::out_of_range("Index out of bounds for dimension " +
                              std::to_string(i));
    }
  }

  int index{tensorData.offset};
  for (std::size_t i{0}; i < getRank(); i++) {
    index += indices[i] * tensorData.strides[i];
  }

  return index;
}

void Tensor::setValue(const Dims& indices, double value) {
  int index = calculateIndex(indices);
  storage->at(index) = value;
}

double Tensor::getValue(const Dims& indices) const {
  int index = calculateIndex(indices);
  return storage->at(index);
}

void Tensor::setValueDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= getNValues()) {
    throw std::out_of_range("Linear index out of bounds");
  }
  storage->at(tensorData.offset + linearIndex) = value;
}

double Tensor::getValueDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= getNValues()) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return storage->at(tensorData.offset + linearIndex);
}

void Tensor::setGradientDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= getNValues()) {
    throw std::out_of_range("Linear index out of bounds");
  }
  gradient->setValueDirect(tensorData.offset + linearIndex, value);
}

double Tensor::getGradientDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= getNValues()) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return gradient->getValueDirect(tensorData.offset + linearIndex);
}

double* Tensor::getData() const { return this->storage->getData(); }

double* Tensor::getGradientData() const { return this->gradient->getData(); }

bool Tensor::getHasGrad() const { return this->gradData.hasGrad; }

void Tensor::setHasGrad(bool hasGrad) { this->gradData.hasGrad = hasGrad; }

Strides Tensor::getStrides() const { return tensorData.strides; }

int Tensor::getOffset() const { return tensorData.offset; }

int Tensor::getNValues() const { return tensorData.dimensions.product(); }

std::size_t Tensor::getRank() const { return tensorData.dimensions.size(); }

Dims Tensor::getDimensions() const { return tensorData.dimensions; }

bool Tensor::getIsLeaf() const { return gradData.isLeaf; }

void Tensor::setLeaf(bool isLeaf) { this->gradData.isLeaf = isLeaf; }

void Tensor::setRequiresGrad(bool requiresGrad) {
  this->gradData.requiresGrad = requiresGrad;
}

bool Tensor::getRequiresGrad() const { return gradData.requiresGrad; }

void Tensor::setGradFunction(std::shared_ptr<GradFunction> gradFunction) {
  this->gradFunction = std::move(gradFunction);
}

Tensor Tensor::detachGradient() {
  if (gradient == nullptr) {
    throw std::invalid_argument("gradient is nullptr");
  }

  Tensor oldGradient = *gradient;

  gradient = std::make_shared<Tensor>(tensorData.dimensions, false);

  if (auto accumulator =
          std::dynamic_pointer_cast<function::GradAccumulator>(gradFunction)) {
    accumulator->setGradient(gradient);
  }

  return oldGradient;
}

std::shared_ptr<GradFunction> Tensor::getGradFunction() {
  return this->gradFunction;
}
}  // namespace mattTorch
