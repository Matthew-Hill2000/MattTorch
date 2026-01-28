#pragma once

#include <mattTorch/function/division/gradDivideScalar.h>
#include <mattTorch/function/gradFunction.h>
#include <mattTorch/function/multiplication/gradMultiplyScalar.h>
#include <mattTorch/tensor/kernels/kernels.h>

#include <memory>
#include <vector>

#include "mattTorch/function/addition/gradAddScalar.h"
#include "mattTorch/function/subtraction/gradSubtractScalar.h"

namespace mattTorch {

namespace tensor {
class TensorStorage;
}
class GradFunction;

class TensorView {
 private:
  // Holds and handles the tensors data
  std::shared_ptr<tensor::TensorStorage> storage;
  // Holds and handles the tensors gradient data
  std::shared_ptr<TensorView> gradient;
  // The function that calculates the gradients of this
  // function with respect to its parents
  std::shared_ptr<GradFunction> gradFunction;

  // Is the tensor a leaf tensor at the edge of the graph (no
  // parents)
  bool isLeaf;
  // Does this tensor require its gradient to be calculated
  // in the backward pass
  bool requiresGrad;
  // Does this tensor currently have a gradient stored in
  // gradFunction
  bool hasGrad;

  // The size of each of the tensors dimensions
  std::vector<int> dimensions;
  // The strides are used to calculate the offset for
  // each dimension
  std::vector<int> strides;
  // the offset is used to define where the elements of a sub
  // tensor start within gradientStorage i.e since gradientStorage
  // is shared
  int offset;
  // the Number of values in the tensor
  int nValues;
  // The number of dimensions
  int rank;

  // Converts a vector of indices into the corresponding index of the
  // appropriate element within the linear storage
  int calculateIndex(const std::vector<int>& indices) const;

 public:
  // Default Constuctor
  TensorView();
  // Constructor to create a tensor of specified dimensionality and whether or
  // not it is a leaf
  TensorView(const std::vector<int>& dims, bool isLeaf = true);
  // Shallow copy of the tensor, i.e creates a tensor that shares the same
  // storage, gradStorage and gradFunction objects
  TensorView(const TensorView& other);
  // Creates a tensor with the specific arguments
  TensorView(std::shared_ptr<tensor::TensorStorage> storage,
             std::vector<int> dims, std::vector<int> strides, int offset,
             int nVals, int rnk, std::shared_ptr<TensorView> gradientStorage,
             std::shared_ptr<GradFunction> gradFunction, bool isLeaf,
             bool requiresGrad, bool hasGrad);

  // Deep copy of the tensor in which the storage, gradStorage and gradFunction
  // are used to make complete copies for the new tensor
  TensorView deepCopy() const;

  // When called on a tensor it starts by calling the backward function of the
  // gradFunction object associated with this tensor, which in turn iteratively
  // calls the backwards functions of the parents of this tensor in the
  // computational graph, iterativelly passing backwards the gradient and
  // calculating succesive gradients of this tensor with respect to the tensors
  // in the graph using the chain rule. The input gradient is used to
  // initialise/seed the gradient and should be set to 1.0 by default
  void backward(TensorView& inputGradient, bool higherDerivatve = false);
  // Adds the value of inputGradient to the value stored in this tensors
  // gradStorage.
  void addGradient(TensorView& inputGradient);
  void resetGradient();

  // Shallow copy of the other tensor, i.e creates a tensor that shares the same
  // storage, gradStorage and gradFunction objects
  TensorView& operator=(const TensorView& other);
  // Sets all of the values in the tensor to the same double value
  TensorView& operator=(double val);

  // Set the value of an element of this tensor at the specified indices
  void setValue(const std::vector<int>& indices, double value);
  // Get the value of an element of this tensor at the specified indices
  double getValue(const std::vector<int>& indices) const;
  // Directly set the value of an element in this tensor by specifying the
  // linear index value directly
  void setValueDirect(int index, double value);
  // Directly get the value of an element in this tensor by specifying the
  // linear index value directly
  double getValueDirect(int index) const;
  // Directly set the value of an element of this tensors gradient by specifying
  // the linear index value directly
  void setGradientDirect(int index, double value);
  // Directly get the value of an element of this tensors gradient by specifying
  // the linear index value directly
  double getGradientDirect(int index) const;

  // Creates a new TensorView object indexed via the first dimension that shares
  // the storage with the original tensor. Values of this tensor are
  // differentiatied from the storage via the calculation of an offset that
  // defines were the values associated with this subtensor begin in memory
  TensorView operator[](int index);
  const TensorView operator[](int index) const;
  // Returns the value of a specific element within the tensor at the specified
  // indices
  double& operator[](const std::vector<int>& indices);
  const double& operator[](const std::vector<int>& indices) const;

  // Creates a new tensor from the elementwise summation of two tensors. Unless
  // neither of the parent tensors require gradients to be calculated for them,
  // the resulting tensor points to a GradAdd object that contains shallow
  // copies of the each tensor that was used in the summation as well as
  // the GradFunction objects associated with those tensors.
  TensorView operator+(const TensorView& other);
  // Creates a new tensor from the elementwise subtraction of two tensors.
  // Unless neither of the parent tensors require gradients to be calculated for
  // them, the resulting tensor points to a gradSubtract object that contains
  // shallow copies of the each tensor that was used in the subtraction as well
  // as the GradFunction objects associated with those tensors.
  TensorView operator-(const TensorView& other);
  // Creates a new tensor from the elementwise multiplication of two tensors.
  // Unless neither of the parent tensors require gradients to be calculated for
  // them, the resulting tensor points to a GradMultiply object that contains
  // shallow copies of the each tensor that was used in the multiplication as
  // well as the GradFunction objects associated with those tensors.
  TensorView operator*(const TensorView& other);
  // Creates a new tensor from the elementwise division of two tensors. Unless
  // neither of the parent tensors require gradients to be calculated for them,
  // the resulting tensor points to a GradDivide object that contains shallow
  // copies of the each tensor that was used in the division as well as
  // the GradFunction objects associated with those tensors.
  TensorView operator/(const TensorView& other);

  // Creates a new tensor by adding to all of the elements in the operated on
  // tensor the value of double scalar. Unless the parent tensor doesnt
  // require a gradient to be calculated, the resulting tensor points to a
  // GradAddScalar object that remembers the value of double scalar to be used
  // in the gradient calculation as well as the GradFunction objects associated
  // with the tensors.
  TensorView operator+(double scalar);
  // Creates a new tensor by subtracting from all of the elements in the
  // operated on tensor the value of double scalar. Unless the parent tensor
  // doesnt require a gradient to be calculated, the resulting tensor points to
  // a GradSubtractScalar object that remembers the value of double scalar to be
  // used in the gradient calculation as well as the GradFunction objects
  // associated with the tensors.
  TensorView operator-(double scalar);
  // Creates a new tensor by multiplying all of the elements in the operated on
  // tensor the value of double scalar. Unless the parent tensor doesnt
  // require a gradient to be calculated, the resulting tensor points to a
  // GradMultiplyScalar object that remembers the value of double scalar to be
  // used in the gradient calculation as well as the GradFunction objects
  // associated with the tensors.
  TensorView operator*(double scalar);
  // Creates a new tensor by dividing all of the elements in the operated on
  // tensor the value of double scalar. Unless the parent tensor doesnt require
  // a gradient to be calculated, the resulting tensor points to a
  // GradDivideScalar object that remembers the value of double scalar to be
  // used in the gradient calculation as well as the GradFunction objects
  // associated with the tensors.
  TensorView operator/(double scalar);

  TensorView elementwiseExponent(int scalar);

  TensorView tanh();
  TensorView exponential();
  TensorView reductionSum(int dim);
  TensorView log();
  TensorView mean();

  // Creates a new tensor from the matrix multiplication of two tensors. Unless
  // neither of the parent tensors require gradients to be calculated for them,
  // the resulting tensor points to a GradMultiplyMatrix object that contains
  // shallow copies of the each tensor that was used in the matrix
  // multiplication as well as the GradFunction objects associated with those
  // tensors.
  TensorView matrixMultiply(const TensorView& other);
  TensorView transposeMultiply(const TensorView& other, bool transposeFirst);
  TensorView ReLU();
  TensorView broadcast(int pos, int dim);

  TensorView& operator+=(const TensorView& other);
  TensorView& operator-=(const TensorView& other);
  TensorView& operator*=(const TensorView& other);
  TensorView& operator/=(const TensorView& other);

  TensorView& operator+=(double scalar);
  TensorView& operator-=(double scalar);
  TensorView& operator*=(double scalar);
  TensorView& operator/=(double scalar);
  // Checks if the values within each of the tensors are within the tolerance of
  // 1e-9 of being equal to each other in every element
  bool operator==(const TensorView& other) const;
  // Checks if the values within each of the tensors arent within the tolerance
  // of 1e-9 of being equal to each other
  bool operator!=(const TensorView& other) const;

  // prints the value of the tensor to the console.
  void print(std::ostream& out) const;
  void printMatrix(std::ostream& out, int spaces) const;
  void printTripleMatrix(std::ostream& out) const;
  void printQuadrupleMatrix(std::ostream& out) const;
  friend std::ostream& operator<<(std::ostream& out, const TensorView& tensor);

  // Getters and Setters for the class attributes

  void setHasGrad(bool hasGrad);
  bool getHasGrad();
  double* getData() const;
  double* getGradientData() const;
  std::vector<int> getStrides() const;
  int getOffset() const;
  int getNValues() const;
  int getRank() const;
  std::vector<int> getDimensions() const;
  TensorView detachGradient();
  bool getIsLeaf() const;
  bool getRequiresGrad() const;
  void setLeaf(bool isLeaf);
  void setRequiresGrad(bool requiresGrad);
  void setGradFunction(std::shared_ptr<GradFunction> gradFunction);
  std::shared_ptr<GradFunction> getGradFunction();
};

inline TensorView operator+(double scalar, TensorView& tensor) {
  TensorView result(tensor.getDimensions(), false);

  // Calculate the result of the mathematical operation and set the value in the
  // resulting tensor.

  tensor::kernels::cpu::tensorScalarAdd(tensor.getData(), scalar,
                                        result.getData(), result.getNValues());
  // If neither parent tensors require gradient then neither does the child,
  // otherwise create the GradFunction object for this mathematical operation
  // with pointers to the tensors needed for the gradient calculation as well as
  // pointers to the GradFunction objects of the parent Tensors
  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(
        std::make_shared<function::GradAddScalar>(scalar, nextFunctions));
  }

  return result;
}
inline TensorView operator-(double scalar, TensorView& tensor) {
  TensorView result(tensor.getDimensions(), false);

  // Calculate the result of the mathematical operation and set the value in the
  // resulting tensor.

  tensor::kernels::cpu::scalarTensorSubtract(
      tensor.getData(), scalar, result.getData(), result.getNValues());
  // If neither parent tensors require gradient then neither does the child,
  // otherwise create the GradFunction object for this mathematical operation
  // with pointers to the tensors needed for the gradient calculation as well as
  // pointers to the GradFunction objects of the parent Tensors
  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(
        std::make_shared<function::GradSubtractScalar>(scalar, nextFunctions));
  }

  return result;
}
inline TensorView operator*(double scalar, TensorView& tensor) {
  TensorView result(tensor.getDimensions(), false);

  // Calculate the result of the mathematical operation and set the value in the
  // resulting tensor.

  tensor::kernels::cpu::tensorScalarMultiplication(
      tensor.getData(), scalar, result.getData(), result.getNValues());
  // If neither parent tensors require gradient then neither does the child,
  // otherwise create the GradFunction object for this mathematical operation
  // with pointers to the tensors needed for the gradient calculation as well as
  // pointers to the GradFunction objects of the parent Tensors
  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(
        std::make_shared<function::GradMultiplyScalar>(scalar, nextFunctions));
  }

  return result;
}
inline TensorView operator/(double scalar, TensorView& tensor) {
  TensorView result(tensor.getDimensions(), false);

  // Calculate the result of the mathematical operation and set the value in the
  // resulting tensor.

  tensor::kernels::cpu::scalarTensorDivision(
      tensor.getData(), scalar, result.getData(), result.getNValues());

  // If neither parent tensors require gradient then neither does the child,
  // otherwise create the GradFunction object for this mathematical operation
  // with pointers to the tensors needed for the gradient calculation as well as
  // pointers to the GradFunction objects of the parent Tensors
  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(std::make_shared<function::GradDivideScalar>(
        scalar, std::make_shared<TensorView>(tensor), nextFunctions, false));
  }

  return result;
}
}  // namespace mattTorch
