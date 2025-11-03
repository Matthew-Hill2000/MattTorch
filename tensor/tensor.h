#ifndef TENSOR_STORAGE_H
#define TENSOR_STORAGE_H

#include <iostream>
#include <memory>
#include <vector>

#include "../tensorView/tensorView.h"

class Function;
class TensorStorage;
class TensorView;

class Tensor {
 private:
  // The actual value of the tensor
  TensorView data;
  // The accumulated value of the gradient of the tensor. It starts out as zero
  // when the tensor is created.
  TensorView gradient;
  // Stores pointers to the tensors that this tensor is used to create.
  std::vector<Tensor*> children;
  // A pointer to the function object that was used to create this tensor. The
  // function object has the parents of this tensor stored as pointers and has
  // the function to calculate the derivatives of the parents given this tensors
  // gradient.
  Function* parentFunction;

 public:
  // Default Constructor
  Tensor();
  // Creates a tensor object with the specificed dimension and parents. uses the
  // dimensions to call the constructor for the TensorView object to create the
  // TensorView Data object for this Tensor.
  Tensor(const std::vector<int>& dims, std::vector<Tensor*> children);
  // Creates a Tensor object with the specific dimensions. uses the dimensions
  // to call the constructor for the TensorView object to create the TensorView
  // Data object for this Tensor. Leaves the parents unset
  Tensor(const std::vector<int>& dims);
  // Copy constructor that just sets all of the values of this tensor to that of
  // the other tensor
  Tensor(const Tensor& other);
  // Creates a Tensor object by directly specifying all of its member variables
  // and the member variables of its TensorView Data object. Sets the parents
  // explicitly.
  Tensor(std::shared_ptr<TensorStorage> storage, std::vector<int> dims,
         std::vector<int> strides, int offset, int nVals, int rnk,
         std::vector<Tensor*> children);

  // Sets this tensor to be the same as the 'other' tensor
  Tensor& operator=(const Tensor& rOther);
  // Sets all of the values in this tensor to 'double val'
  Tensor& operator=(double val);

  // Set the value in this tensor at the position specified by 'indices' to
  // 'value'. this function just acts as a wrapper of the TensorView function
  // with the same name, calling it to change the value in this Tensor objets
  // data attribute.
  void setValue(const std::vector<int>& indices, double value);
  // Get the value in this tensor at the position specified by 'indices' to
  // 'value'. This function just acts as a wrapper of the TensorView function
  // with the same name, calling it to get the value in this Tensor objets data
  // attribute.
  double getValue(const std::vector<int>& indices) const;
  // Set the value in the tensorStorage of the data Tensorview object of this
  // Tensor at the position specified by index to 'value'. This function just
  // acts as a wrapper of the TensorView function with the same name, calling it
  // to change the value.
  void setValueDirect(int index, double value);
  // Get the value in the tensorStorage of the data Tensorview object of this
  // Tensor at the position specified by index. this function just
  // acts as a wrapper of the TensorView function with the same name, calling it
  // to get the value in this Tensor objets data attribute.
  double getValueDirect(int index) const;

  // Itteratively calls the backward method associated with the parentFunction
  // of this object and each of the objects upstream in the computational graph
  void backward();

  Tensor operator+(const Tensor& rOther) const;
  Tensor operator-(const Tensor& rOther) const;
  Tensor operator*(const Tensor& rOther) const;
  Tensor operator/(const Tensor& rOther) const;
  Tensor getData();

  void setBackwardFunction(Function backwardFunction);

  Tensor operator[](int index);
  const Tensor operator[](int index) const;
  double& operator[](const std::vector<int>& rIndices);
  const double& operator[](const std::vector<int>& rIndices) const;
};
#endif
