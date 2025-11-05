#include "tensor.h"

#include "../function/accumulator/gradAccumulator.h"
#include "../function/addition/gradAdd.h"
#include "../function/division/gradDivide.h"
#include "../function/exponent/gradExponent.h"
#include "../function/multiply/gradMultiply.h"
#include "../function/subtraction/gradSubtract.h"

// Constructors
///////////////////////////////////////////////////////////////////////////////
Tensor::Tensor(const TensorView& data, bool requiresGrad = true,
               GradFunction* gradFunction)
    : data{data}, requiresGrad{requiresGrad}, gradFunction{gradFunction} {}

Tensor::Tensor(const std::vector<int>& dims, bool requiresGrad = true)
    : requiresGrad{requiresGrad} {
  this->data = TensorView(dims);
  this->gradFunction = new GradAccumulator(this);
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
  this->gradient += inputGradient.getData();
}

// Operator Overloads for Tensor-Tensor maths
/////////////////////////////////////////////////////////////////////////////////
Tensor Tensor::operator+(const Tensor& other) const {
  TensorView outputData = this->data + other.data;

  GradAdd* gradFunction = new GradAdd(
      std::vector<Tensor*>{this, &other},
      std::vector<GradFunction*>{this->gradFunction, other.gradFuncton});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator-(const Tensor& other) const {
  TensorView outputData = this->data - other.data;

  GradSubtract* gradFunction =
      new GradSubtract(std::vector<Tensor*>{this, &other});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator*(const Tensor& other) const {
  TensorView outputData = this->data * other.data;

  GradMultiply* gradFunction =
      new GradMultiply(std::vector<Tensor*>{this, &other});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator/(const Tensor& other) const {
  TensorView outputData = this->data - other.data;

  GradDivide* gradFunction = new GradDivide(std::vector<Tensor*>{this, &other});

  Tensor output(outputData, gradFunction);
  return output
}

Tensor Tensor::exponent(int exponent) {
  TensorView outputData = this->data;

  for (int i{1}; i < exponent; i++) {
    outputData *= this->data;
  }

  GradExponent* gradFunction =
      new GradExponent(std::vector<Tensor*>{this}, exponent);

  Tensor output(outputData, gradFunction);
  return output;
}

// Operator overloads for Tensor-double maths
///////////////////////////////////////////////////////////////////////////////
Tensor Tensor::operator+(double scalar) const {
  TensorView outputData = this->data + scalar;

  GradAddScalar* gradFunction =
      new GradAddScalar(std::vector<Tensor*>{this, &other},
                        std::vector<GradFunction*>{this->gradFunction});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator-(double scalar) const {
  TensorView outputData = this->data - scalar;

  GradSubtractScalar* gradFunction =
      new GradSubtractScalar(std::vector<Tensor*>{this});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator*(double scalar) const {
  TensorView outputData = this->data * scalar;

  GradMultiplyScalar* gradFunction =
      new GradMultiplyScalar(std::vector<Tensor*>{this});

  Tensor output(outputData, gradFunction);
  return output;
}

Tensor Tensor::operator/(double scalar) const {
  TensorView outputData = this->data / scalar;

  GradDivideScalar* gradFunction =
      new GradDivideScalar(std::vector<Tensor*>{this, &other});

  Tensor output(outputData, gradFunction);
  return output;
}

// Getters and Setters for member attributes
///////////////////////////////////////////////////////////////////////////////
TensorView Tensor::getData{return this->data};
TensorView Tensor::getGradient { return this->gradient; }
Function* getGradFunction { return this->gradFunction; }
bool getIsLeaf { return this->isLeaf; }
bool getRequiresGrad { return this->requiresGrad; }
