#include "tensor.h"

#include "../function/accumulator/gradAccumulator.h"
#include "../function/addition/gradAdd.h"
#include "../function/addition/gradAddScalar.h"
#include "../function/division/gradDivide.h"
#include "../function/division/gradDivideScalar.h"
#include "../function/exponent/gradExponent.h"
#include "../function/multiply/gradMultiply.h"
#include "../function/multiply/gradMultiplyScalar.h"
#include "../function/subtraction/gradSubtract.h"
#include "../function/subtraction/gradSubtractScalar.h"

// Constructors
///////////////////////////////////////////////////////////////////////////////
Tensor::Tensor(const TensorView& data, bool requiresGrad, bool isLeaf,
               GradFunction* gradFunction)
    : data{data},
      requiresGrad{requiresGrad},
      isLeaf{isLeaf},
      gradFunction{gradFunction} {}

Tensor::Tensor(const std::vector<int>& dims, bool requiresGrad, bool isLeaf)
    : requiresGrad{requiresGrad} {
  this->data = TensorView(dims);
  this->gradFunction = new GradAccumulator(this);
  this->isLeaf = true;
}

Tensor::Tensor(const Tensor& other)
    : data{other.data},
      gradient{other.gradient},
      gradFunction{other.gradFunction},
      isLeaf{other.isLeaf},
      requiresGrad{other.requiresGrad} {}

// Assignment Operators
/////////////////////////////////////////////////////////////////////////////////
Tensor& Tensor::operator=(const Tensor& other) {
  if (this != &other) {
    this->data = other.data;
    this->gradient = other.gradient;
    this->gradFunction = other.gradFunction;
    this->isLeaf = other.isLeaf;
    this->requiresGrad = other.requiresGrad;
  }
  return *this;
}

Tensor& Tensor::operator=(double val) {
  this->data = val;
  return *this;
}

// Set Value / Get Value Functions
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

// Functions for computational graph algorithm
///////////////////////////////////////////////////////////////////////////////
void Tensor::backward(Tensor& inputGradient) {
  this->gradFunction->backward(inputGradient);
}

void Tensor::addGradient(Tensor& inputGradient) {
  if (this->hasGrad) {
    this->gradient = inputGradient.getData();
  } else {
    this->gradient += inputGradient.getData();
  }
}

// Operator Overloads for Tensor-Tensor maths
/////////////////////////////////////////////////////////////////////////////////
Tensor Tensor::operator+(const Tensor& other) {
  TensorView outputData = this->data + other.data;

  GradAdd* gradFunction = new GradAdd(
      std::vector<const Tensor*>{this, &other},
      std::vector<GradFunction*>{this->gradFunction, other.gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator-(const Tensor& other) {
  TensorView outputData = this->data - other.data;

  GradSubtract* gradFunction = new GradSubtract(
      std::vector<const Tensor*>{this, &other},
      std::vector<GradFunction*>{this->gradFunction, other.gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator*(const Tensor& other) {
  TensorView outputData = this->data * other.data;

  GradMultiply* gradFunction = new GradMultiply(
      std::vector<const Tensor*>{this, &other},
      std::vector<GradFunction*>{this->gradFunction, other.gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator/(const Tensor& other) {
  TensorView outputData = this->data - other.data;

  GradDivide* gradFunction = new GradDivide(
      std::vector<const Tensor*>{this, &other},
      std::vector<GradFunction*>{this->gradFunction, other.gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::exponent(int exponent) {
  TensorView outputData = this->data;

  for (int i{1}; i < exponent; i++) {
    outputData *= this->data;
  }

  GradExponent* gradFunction = new GradExponent(
      std::vector<const Tensor*>{this},
      std::vector<GradFunction*>{this->gradFunction}, exponent);

  Tensor output(outputData, gradFunction);
  return output;
}

// Operator overloads for Tensor-double maths
///////////////////////////////////////////////////////////////////////////////
Tensor Tensor::operator+(double scalar) {
  TensorView outputData = this->data + scalar;

  GradAddScalar* gradFunction =
      new GradAddScalar(std::vector<const Tensor*>{this},
                        std::vector<GradFunction*>{this->gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator-(double scalar) {
  TensorView outputData = this->data - scalar;

  GradSubtractScalar* gradFunction =
      new GradSubtractScalar(std::vector<const Tensor*>{this},
                             std::vector<GradFunction*>{this->gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator*(double scalar) {
  TensorView outputData = this->data * scalar;

  GradMultiplyScalar* gradFunction =
      new GradMultiplyScalar(std::vector<const Tensor*>{this},
                             std::vector<GradFunction*>{this->gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator/(double scalar) {
  TensorView outputData = this->data / scalar;

  GradDivideScalar* gradFunction =
      new GradDivideScalar(std::vector<const Tensor*>{this},
                           std::vector<GradFunction*>{this->gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

// Getters and Setters for member attributes
///////////////////////////////////////////////////////////////////////////////
TensorView Tensor::getData() const { return this->data; }
TensorView Tensor::getGradient() const { return this->gradient; }
GradFunction* Tensor::getGradFunction() const { return this->gradFunction; }
bool Tensor::getIsLeaf() const { return this->isLeaf; }
bool Tensor::getRequiresGrad() const { return this->requiresGrad; }
