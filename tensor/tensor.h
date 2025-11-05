#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <iostream>
#include <memory>
#include <vector>

#include "../function/gradFunction.h"
#include "../tensorView/tensorView.h"

class Function;
class TensorStorage;
class TensorView;

class Tensor {
 private:
  TensorView data;             // Value of the tensor
  TensorView gradient;         // Value of the gradient of the tensor
  GradFunction* gradFunction;  // pointer to the functor to calculate gradient
  bool isLeaf;
  bool requiresGrad;

 public:
  // Constructors
  /////////////////////////////////////////////////////////////////////////////
  Tensor();
  Tensor(const TensorView& data, bool requiresGrad = true,
         GradFunction* gradFunction = nullptr);
  Tensor(const std::vector<int>& dims, bool requiresGrad = true);
  Tensor(const Tensor& other);

  // Assignment Operators
  /////////////////////////////////////////////////////////////////////////////
  Tensor& operator=(const Tensor& rOther);
  Tensor& operator=(double val);

  // Set Value / Get Value Functions
  /////////////////////////////////////////////////////////////////////////////
  void setValue(const std::vector<int>& indices, double value);
  double getValue(const std::vector<int>& indices) const;
  void setValueDirect(int index, double value);
  double getValueDirect(int index) const;

  // Functions for computational graph algorithm
  /////////////////////////////////////////////////////////////////////////////
  void backward(Tensor& inputGradient);
  void addGradient(Tensor& inputGradient);

  // Operator Overloads for Tensor-Tensor maths
  /////////////////////////////////////////////////////////////////////////////
  Tensor operator+(const Tensor& rOther) const;
  Tensor operator-(const Tensor& rOther) const;
  Tensor operator*(const Tensor& rOther) const;
  Tensor operator/(const Tensor& rOther) const;

  Tensor& operator+=(const Tensor& rOther);
  Tensor& operator-=(const Tensor& rOther);
  Tensor& operator*=(const Tensor& rOther);
  Tensor& operator/=(const Tensor& rOther);

  Tensor exponent(int exponent);

  // Operator overloads for Tensor-scalar maths
  /////////////////////////////////////////////////////////////////////////////
  Tensor operator+(double scalar) const;
  Tensor operator-(double scalar) const;
  Tensor operator*(double scalar) const;
  Tensor operator/(double scalar) const;

  Tensor& operator+=(double scalar);
  Tensor& operator-=(double scalar);
  Tensor& operator*=(double scalar);
  Tensor& operator/=(double scalar);

  // Getters and Setters for member attributes
  /////////////////////////////////////////////////////////////////////////////
  TensorView getData();
  TensorView getGradient();
  Function* getGradFunction();
  bool getIsLeaf();
  bool getRequiresGrad();

  void setLeaf(bool leaf);
  void setRequiresGrad();

  // Overloads for indexing of data values
  /////////////////////////////////////////////////////////////////////////////
  Tensor operator[](int index);
  const Tensor operator[](int index) const;
  double& operator[](const std::vector<int>& rIndices);
  const double& operator[](const std::vector<int>& rIndices) const;
};
#endif
