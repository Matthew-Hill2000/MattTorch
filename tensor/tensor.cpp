#include "tensor.h"

#include "../function/function.h"
#include "../function/multiply/multiply.h"

Tensor::Tensor(const std::vector<int>& dims, std::vector<Tensor*> children)
    : children{children} {
  this->data = TensorView(dims);
}

Tensor::Tensor(const std::vector<int>& dims) { data = TensorView(dims); }

Tensor::Tensor(const Tensor& other)
    : data{other.data},
      gradient{other.gradient},
      children{other.children},
      parentFunction{other.parentFunction} {}

Tensor::Tensor(std::shared_ptr<TensorStorage> storage, std::vector<int> dims,
               std::vector<int> strides, int offset, int nVals, int rnk,
               std::vector<Tensor*> children) {
  this->data = TensorView(storage, dims, strides, offset, nVals, rnk);
  this->children = children;
}

/////////////////////////////////////////////////////////////////////////////////

Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    this->data = other.data;
    this->gradient = other.gradient;
    this->parentFunction = other.parentFunction;
    this->children = other.children;
  }
  return *this;
}

Tensor& Tensor::operator=(double val) {
  this->data = val;
  return *this;
}

/////////////////////////////////////////////////////////////////////////////////

void Tensor::setValue(const std::vector<int>& indices, double value) {
  this->data.setValue(indices, value);
}

double Tensor::getValue(const std::vector<int>& indices) const {
  return this->data.getValue(indices);
}
void Tensor::setValueDirect(int index, double value) {
  this->data.setValueDirect(index, value);
}
double Tensor::getValueDirect(int index) const {
  return this->data.getValueDirect(index);
}

/////////////////////////////////////////////////////////////////////////////////

Tensor Tensor::operator+(const Tensor& other) const {
  TensorView outData = this->data + other.data;
  std::vector<Tensor*> parents = {this, &other};

  Tensor out(outData.getDimensions(), children);

  out.data = outData;
  out.parentFunction = new MultiplyFunction(parents, &out);

  return out;
}

Tensor Tensor::operator-(const Tensor& other) const {}
Tensor Tensor::operator*(const Tensor& other) const {}
Tensor Tensor::operator/(const Tensor& other) const {}
