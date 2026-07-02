#pragma once

#include <mattTorch/tensor/kernels/kernels.h>

#include <cstddef>
#include <memory>

#include "tensorTypes.h"

namespace mattTorch {

namespace tensor {
class TensorStorage;
}

class GradFunction;

/**
 * @brief An arbitrary-dimension tensor with automatic differentiation support.
 *
 * The Tensor class is the foundation for all of MattTorch. Tensor
 * objects can be created with any arbitrary number of dimensions with any
 * arbitrary sizes. When creating a Tensor object, one specifies the
 * dimensions via a Dims object of dimension sizes.
 *
 * The data held within a Tensor object is handled by an instance of the
 * TensorStorage class, pointed to by the storage attribute of the Tensor
 * class in the form of a std::shared_ptr.
 *
 * Mathematical operations on the Tensor class are overloaded to allow for
 * automatic differentiation to be carried out on any resulting tensors, after
 * any number of chained or combined mathematical opeartions. This is
 * facilitated by the construction of a computational graph during the
 * application of successive mathematical operations, specifically a directed
 * acyclical graph of GradFunction objects, each consisting of a backward method
 * to calculate the local derivative of its associated operation, and passing
 * the derivative to its parent nodes within the graph allows for the total
 * derivative to be computed via the chain rule.
 *
 * @see TensorStorage for the class that handles the continguous data storage
 * @see GradFunction for the base class of differentiable operations.
 * @see backward() for gradient computation.
 */
class Tensor {
 private:
  /// A shared pointer to a TensorStorage object, tasked with holding and
  /// handling the tensors data as a continguous block of memory
  std::shared_ptr<tensor::TensorStorage> storage;

  /// Holds and handles the tensors gradient data as a seperate Tensor
  /// object
  std::shared_ptr<Tensor> gradient;

  // The GradFunction object that represents a node in a computational graph
  // associated with the creation of this tensor, the specific form of
  // GradFunction object assigned to this tensor matches the operation used to
  // creat it, and therefore contains the correct method with which gradients
  // can be calculated for this Tensor with respect to its parents in the
  // computational graph. If this is a leaf tensor that requires gradient
  // computation, the GradFunction object is set to that of a GradAccumulator,
  // that handles accumulation of gradients into this Tensor objects
  // gradient Tensor attribute.
  std::shared_ptr<GradFunction> gradFunction;

  // The metadata associated with interpreting the flat, contiguous
  // memory of a Tensor as a multidimensional array.
  //
  // The dimensions attribute is a vector of integers used to store the size of
  // each dimension of a Tensor.
  //
  // The strides attribute is a vector of integers used to store how many
  // elements one must step through the contiguous memory of a Tensor in order
  // to increment a specific multi-dimensional index by one.
  //
  // The offset attribute is used by sub tensors to define where in the
  // underlying contiguous storage of values do the values for this sub tensors
  // begin.
  TensorData tensorData;

  // The metadata associated with the construction and subsequent traversal
  // of the computational graph that is utilised in the computation of
  // gradients.
  //
  // The isLeaf attribute is a boolean value that describes whether this tensor
  // sits at the end of a computational graph
  //
  // The requiresGrad attribute is a boolean that states whether this Tensor
  // shall require gradients being calculated with respect to itself in the
  // backward pass of any associated computational graphs.
  //
  // The hasGrad attribute is a boolean value that states whether the associated
  // Tensor has any gradient values so far computed and stored within its
  // .gradient attribute
  GradData gradData;

  /**
   * @brief Helper function to calculate the index along the contiguous data
   * storage for this Tensor associated with a specific set of indices
   *
   *
   * @param indices The indices with which the contiguous data index
   * should be calculated for
   *
   * @return An integer representing the index along the contiguous data storage
   * at which the element associated with the input dimensions shall be found
   */
  int calculateIndex(const Dims& indices) const;

 public:
  /**
   * @brief Default constructor creating an empty Tensor
   */
  Tensor();

  /**
   * @brief Constructor to create a Tensor of specified dimensionality
   *
   * @param dims A vector specifying the size of each dimension
   * @param isLeaf Whether this tensor is a leaf node, sitting at the end of the
   * computational graph, defaults to true
   */
  Tensor(const Dims& dims, bool isLeaf = true);

  /**
   * @brief Shallow copy constructor
   *
   * Creates a Tensor that shares the same storage, gradient and
   * gradFunction objects as the original tensor. Modifications to the
   * underlying data of either tensor will affect both.
   *
   * @param other The Tensor to create a shallow copy from
   */
  Tensor(const Tensor& other) = default;

  /**
   * @brief Full constructor with explicit specification of all attributes
   *
   * Creates a Tensor with complete control over all internal attributes. This
   * constructor is primarily used internally when creating tensors as the
   * result of mathematical operations.
   *
   * @param storage Shared pointer to the TensorStorage holding the data
   * @param tensorData metaData associated with the dimensions, strides and
   * offset into the underlying storage
   * @param gradientStorage Shared pointer to the Tensor holding gradient
   * data
   * @param gradFunction Shared pointer to the GradFunction for backpropagation
   * @param gradData metadata associated with the state of the tensor in terms
   * of its status within any computational graph calculations
   */
  Tensor(std::shared_ptr<tensor::TensorStorage> storage, TensorData tensorData,
         std::shared_ptr<Tensor> gradientStorage,
         std::shared_ptr<GradFunction> gradFunction, GradData gradData);

  /*
   * @brief The move assignment operator gives ownership of the moved from
   * Tensors attributes to the moved to Tensor.
   *
   * The attributes of the moved from tensor have their ownership transferred to
   * the moved to Tensor. The moved from Tensor is left in a moved from state in
   * which its pointers are null and other attributes empty.
   *
   * @param other The Tensor from which to move ownership from
   * @return The Tensor to which ownership is moved to
   */
  Tensor& operator=(Tensor&& other) = default;

  /*
   * @brief The move constructor gives ownership of the moved from
   * Tensors attributes to the moved to Tensor.
   *
   * The attributes of the moved from tensor have their ownership transferred to
   * the moved to Tensor. The moved from Tensor is left in a moved from state in
   * which its pointers are null and other attributes empty.
   *
   * @param other The Tensor from which to move ownership from
   */
  Tensor(Tensor&& other) = default;

  /*
   * @brief The destructor for the Tensor class
   *
   * The destructor for the Tensor class is defaulted. The destructor calls the
   * default destructor of each of the Tensors attributes. This allows the
   * shared pointers to appropriately handle updating their reference counters
   * and such.
   */
  ~Tensor() = default;

  /**
   * @brief performs a deep copy of the Tensor object, in which all of the
   * values within its storage are copied into the resulting Tensors storage
   *
   * @return a Tensor object with a deep copy of the original Tensors
   * storage
   */
  Tensor deepCopy() const;

  /**
   * @brief Performs automatic differentiation to calculate and accumulate the
   * gradients of all of the leaf tensors within the computation graph that
   * culminates at this Tensor, which contributed towards the evaluation of this
   * Tensor
   *
   * Initiates the backwards traversal of the computational graph in order to
   * calculate gradients of all of the leaf tensors that contributed to the
   * evaluation of this Tensor. The input gradient, for most cases, should
   * be set to a Tensor with all elements set to 1.0, of the same shape as
   * this Tensor, to represent the initial gradient of df/df = 1.0 (this
   * Tensor == f). This method calls the backward method of the GradFunction
   * object associated to this Tensor, which subsequently calls the
   * backwards functions of the GradFunction objects of its parents within the
   * computational graph, passing to each the derivative of this tensor with
   * respect to each parent respectively
   *
   * @param inputGradient The gradient of this Tensor with respect to itself
   * df/df. Should be 1.0 for standard gradient computation.
   *
   * @param higherDerivatve A bool that represents if higher order derivatives
   * need to be calculated for this Tensor with respect to any dependent
   * Tensors. If True, the mathematical operations performed throughout the
   * backward pass create a new seperate computational graph, which ends at
   * df/dx, that can be then travesed backwards from df/dx back to x to get
   * d2f/dx2.
   */
  void backward(Tensor& inputGradient, bool higherDerivatve = false);

  /**
   * @brief Performs automatic differentiation to calculate and accumulate the
   * gradients of all of the leaf tensors within the computation graph, that
   * contributed towards the evaluation of this Tensor
   *
   * Initiates the backwards traversal of the computational graph in order to
   * calculate gradients of all of the leaf tensors that contributed to the
   * evaluation of this Tensor. This method calls the backward method of the
   * GradFunction object associated to this Tensor, which subsequently calls
   * the backwards functions of the GradFunction objects of its parents within
   * the computational graph, passing to each the derivative of this tensor with
   * respect to each parent respectively
   *
   * @param higherDerivatve A bool that represents if higher order derivatives
   * need to be calculated for this Tensor with respect to any dependent
   * Tensors. If True, the mathematical operations performed throughout the
   * backward pass create a new seperate computational graph, which ends at
   * df/dx, that can be then travesed backwards from df/dx back to x to get
   * d2f/dx2.
   */
  void backward(bool higherDerivatve = false);

  /**
   * @brief Resets the gradient of this tensor to zero
   *
   * Reassigns the gradient attribute to a new shared_ptr initialised with a new
   * Tensor object for the gradient storage. The new gradient Tensor
   * then has its requiresGrad attribute set to false. The GradAccumulator
   * associated with this Tensor is then updated to accumulate gradients
   * within the new gradient object.
   *
   */
  void resetGradient();

  /**
   * @brief Shallow copy assignment operator
   *
   * Creates a shallow copy of this Tensor that consequently shares the same
   * storage, gradient and gradFunction.
   *
   * @param other The Tensor to create a shallow copy of
   * @return Reference to this Tensor after assignment
   */
  Tensor& operator=(const Tensor& other) = default;

  /**
   * @brief Scalar assignment operator
   *
   * Sets all of the values in the tensor to the same double value
   *
   * @param val The value to assign to all elements
   * @return Reference to this Tensor after assignment
   */
  Tensor& operator=(double val);

  /**
   * @brief Sets the value of an element at the specified indices
   *
   * Sets the value of an element at the specified indices. Uses the helper
   * function calculateIndex() to convert the Dims indices into the equivalent
   * index for the contiguous data storage.
   *
   * @param indices Dims specifying the element location
   * @param value The value to set at the specified location
   */
  void setValue(const Dims& indices, double value);

  /**
   * @brief Gets the value of an element at the specified indices
   *
   * Returns a copy of the value of an element at the specified indices. Uses
   * the helper function calculateIndex() to convert the Dims indices
   * indices into the equivalent index for the contiguous data storage.
   *
   * @param indices Dims specifying the element location
   * @return The value at the specified location
   */
  double getValue(const Dims& indices) const;

  /**
   * @brief Sets the value within the tensor at the position within its
   * contiguous data storage specified by the index parameter
   *
   * Sets the value at the position within the contiguous data storage indexed
   * by the given index parameter. Bypasses the multi-dimensional indexing
   * system, instead indexing directly the underlying contiguous storage.
   *
   * @param index The linear index into the storage
   * @param value The value to set at the specified index
   */
  void setValueDirect(int index, double value);

  /**
   * @brief Gets the value within the tensor at the position within its
   * contiguous data storage specified by the index parameter
   *
   * Returns a copy of the value at the position within the contiguous data
   * storage indexed by the given index parameter. Bypasses the
   * multi-dimensional indexing system, instead indexing directly the underlying
   * contiguous storage.
   *
   * @param index The linear index into the storage
   * @return The value at the specified index
   */
  double getValueDirect(int index) const;

  /**
   * @brief Sets the value within the gradient Tensor at the position within
   * its contiguous data storage specified by the index parameter
   *
   * Sets the value at the position within the contiguous data storage of this
   * Tensors gradient, indexed by the given index parameter. Bypasses the
   * multi-dimensional indexing system, instead indexing directly the underlying
   * contiguous storage.
   *
   * @param index The linear index into the gradients storage
   * @param value The gradient value to set at the specified index
   */
  void setGradientDirect(int index, double value);

  /**
   * @brief Gets the value within the gradient Tensor at the position within
   * its contiguous data storage specified by the index parameter
   *
   * Returns a copy of the value at the position within the contiguous data
   * storage of this Tensors gradient, indexed by the given index parameter.
   * Bypasses the multi-dimensional indexing system, instead indexing directly
   * the underlying contiguous storage.
   *
   * @param index The linear index into the gradients storage
   * @return The gradient value at the specified index
   */
  double getGradientDirect(int index) const;

  /**
   * @brief Creates a sub-tensor indexed along the first dimension
   *
   * Creates a new Tensor object indexed along the first dimension that
   * shares the storage with the original tensor. Values of this tensor are
   * differentiatied from the storage via the calculation of an offset that
   * defines were the values associated with this subtensor begin within the
   * contigous data storage
   *
   * @param index The index along the first dimension to slice at
   * @return A new Tensor representing the sub-tensor at the given index
   */
  Tensor operator[](int index);

  /**
   * @brief Creates a const sub-tensor indexed along the first dimension
   *
   * Creates a const Tensor object indexed along the first dimension
   * that shares the storage with the original tensor. Values of this tensor are
   * differentiatied from the storage via the calculation of an offset that
   * defines were the values associated with this subtensor begin within the
   * contigous data storage
   *
   * @param index The index along the first dimension to slice at
   * @return A const Tensor representing the sub-tensor at the given index
   */
  Tensor operator[](int index) const;

  /**
   * @brief Returns a reference to the element at the specified indices
   *
   * Returns a reference to the value of an element at the specified indices.
   * Uses the helper function calculateIndex() to convert the Dims indices
   * into the equivalent index for the contiguous data storage.
   *
   * @param indices Dims specifying the element index
   * @return Reference to the value at the specified location
   */
  double& operator[](const Dims& indices);

  /**
   * @brief Returns a const reference to the element at the specified indices
   *
   * Returns a const reference to the value of an element at the specified
   * indices. Uses the helper function calculateIndex() to convert the
   * Dims indices into the equivalent index for the contiguous data storage.
   *
   * @param indices Dims specifying the element index
   * @return Const reference to the value at the specified location
   */
  const double& operator[](const Dims& indices) const;

  /**
   * @brief Elemtwise addition of two Tensors
   *
   * Returns a new Tensor from the elementwise summation of two tensors. If
   * either of the parents require gradient calculation as part of the
   * construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradAdd object. The GradAdd
   * object itself is instantiated with pointers to the GradFunction objects
   * associated with each of the tensors involved in the elementwise addition,
   * as well as shallow copies of the tensors themselves. This is so that during
   * the backward pass, the derivative of the resulting tensor with respect to
   * each of the parent tensors can be computed and passed backward to each
   * respective parent such that the gradients w.r.t all subsequent parents in
   * the computational graph can be iteratively calculated via the chain rule.
   *
   * @param other The tensor to add to this tensor
   * @return A new Tensor containing the elementwise sum
   *
   */
  Tensor operator+(const Tensor& other);

  /**
   * @brief Elemtwise subtraction of two Tensors
   *
   * Returns a new Tensor from the elementwise subtraction of two tensors. If
   * either of the parents require gradient calculation as part of the
   * construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradSubtract object. The
   * GradSubtract object itself is instantiated with pointers to the
   * GradFunction objects associated with each of the tensors involved in the
   * elementwise subtraction, as well as shallow copies of the tensors
   * themselves. This is so that during the backward pass, the derivative of the
   * resulting tensor with respect to each of the parent tensors can be computed
   * and passed backward to each respective parent such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param other The tensor to subtract from this tensor
   * @return A new Tensor containing the elementwise difference
   *
   */
  Tensor operator-(const Tensor& other);

  /**
   * @brief Elemtwise multiplication of two Tensors
   *
   * Returns a new Tensor from the elementwise multiplication of two tensors. If
   * either of the parents require gradient calculation as part of the
   * construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradMultiply object. The
   * GradMultiply object itself is instantiated with pointers to the
   * GradFunction objects associated with each of the tensors involved in the
   * elementwise multiplication, as well as shallow copies of the tensors
   * themselves. This is so that during the backward pass, the derivative of the
   * resulting tensor with respect to each of the parent tensors can be computed
   * and passed backward to each respective parent such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param other The tensor to multiply with this tensor
   * @return A new Tensor containing the elementwise product
   *
   */
  Tensor operator*(const Tensor& other);

  /**
   * @brief Elemtwise division of two Tensors
   *
   * Returns a new Tensor from the elementwise division of two tensors. If
   * either of the parents require gradient calculation as part of the
   * construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradDivide object. The GradDivide
   * object itself is instantiated with pointers to the GradFunction objects
   * associated with each of the tensors involved in the elementwise division,
   * as well as shallow copies of the tensors themselves. This is so that during
   * the backward pass, the derivative of the resulting tensor with respect to
   * each of the parent tensors can be computed and passed backward to each
   * respective parent such that the gradients w.r.t all subsequent parents in
   * the computational graph can be iteratively calculated via the chain rule.
   *
   * @param other The tensor to divide this tensor by
   * @return A new Tensor containing the elementwise quotient
   *
   */
  Tensor operator/(const Tensor& other);

  /**
   * @brief Addition of a scalar to all elements of this Tensor
   *
   * Returns a new Tensor from the addition of a scalar value to all elements
   * of this tensor. If the parent tensor requires gradient calculation as part
   * of the construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradAddScalar object. The
   * GradAddScalar object itself is instantiated with pointers to the
   * GradFunction object associated with the tensor involved in the scalar
   * addition, as well as the value of the scalar. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param scalar The scalar value to add to each element
   * @return A new Tensor with the scalar added to all elements
   *
   */
  Tensor operator+(double scalar);

  /**
   * @brief Subtraction of a scalar from all elements of this Tensor
   *
   * Returns a new Tensor from the subtraction of a scalar value from all
   * elements of this tensor. If the parent tensor requires gradient calculation
   * as part of the construction of a computational graph, then the new tensor
   * has its gradFunction attribute set to point to a GradSubtractScalar object.
   * The GradSubtractScalar object itself is instantiated with pointers to the
   * GradFunction object associated with the tensor involved in the scalar
   * subtraction, as well as the value of the scalar. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param scalar The scalar value to subtract from each element
   * @return A new Tensor with the scalar subtracted from all elements
   *
   */
  Tensor operator-(double scalar);

  /**
   * @brief Multiplication of a scalar with all elements of this Tensor
   *
   * Returns a new Tensor from the multiplication of a scalar value with all
   * elements of this tensor. If the parent tensor requires gradient calculation
   * as part of the construction of a computational graph, then the new tensor
   * has its gradFunction attribute set to point to a GradMultiplyScalar object.
   * The GradMultiplyScalar object itself is instantiated with pointers to the
   * GradFunction object associated with the tensor involved in the scalar
   * multiplication, as well as the value of the scalar. This is so that during
   * the backward pass, the derivative of the resulting tensor with respect to
   * the parent tensor can be computed and passed backward such that the
   * gradients w.r.t all subsequent parents in the computational graph can be
   * iteratively calculated via the chain rule.
   *
   * @param scalar The scalar value to multiply each element by
   * @return A new Tensor with all elements multiplied by the scalar
   *
   */
  Tensor operator*(double scalar);

  /**
   * @brief Division of all elements of this Tensor by a scalar
   *
   * Returns a new Tensor from the division of all elements of this tensor by a
   * scalar value. If the parent tensor requires gradient calculation as part of
   * the construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradDivideScalar object. The
   * GradDivideScalar object itself is instantiated with pointers to the
   * GradFunction object associated with the tensor involved in the scalar
   * division, as well as the value of the scalar. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param scalar The scalar value to divide each element by
   * @return A new Tensor with all elements divided by the scalar
   *
   */
  Tensor operator/(double scalar);

  /**
   * @brief Elemtwise exponentiation of a Tensor by an integer scalar
   *
   * Returns a new Tensor from the elementwise exponentiation of all elements
   * of this tensor by an integer scalar. If the parent tensor requires gradient
   * calculation as part of the construction of a computational graph, then the
   * new tensor has its gradFunction attribute set to point to a GradExponent
   * object. The GradExponent object itself is instantiated with a pointer to
   * the GradFunction object associated with the tensor involved in the
   * elementwise exponentiation, as well as the value of the integer exponent
   * and a shallow copy of the tensor itself. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param scalar The integer exponent to raise each element to
   * @return A new Tensor with each element raised to the given power
   *
   */
  Tensor elementwiseExponent(int scalar);

  /**
   * @brief Elemtwise hyperbolic tangent of a Tensor
   *
   * Returns a new Tensor from the elementwise application of the hyperbolic
   * tangent to all elements of this tensor. If the parent tensor requires
   * gradient calculation as part of the construction of a computational graph,
   * then the new tensor has its gradFunction attribute set to point to a
   * GradTanh object. The GradTanh object itself is instantiated with a
   * pointer to the GradFunction object associated with the tensor with which
   * the operation of Tanh was applied to and a shallow copy of the tensor
   * itself. This is so that during the backward pass, the derivative of the
   * resulting tensor with respect to the parent tensor can be computed and
   * passed backward such that the gradients w.r.t all subsequent parents in the
   * computational graph can be iteratively calculated via the chain rule.
   *
   * @return A new Tensor with tanh applied to all elements
   *
   */
  Tensor tanh();

  /**
   * @brief Elemtwise exponential of a Tensor
   *
   * Returns a new Tensor from the elementwise application of the eponential
   * function to all elements of this tensor. If the parent tensor requires
   * gradient calculation as part of the construction of a computational graph,
   * then the new tensor has its gradFunction attribute set to point to a
   * GradExponential object. The GradExponential object itself is instantiated
   * with a pointer to the GradFunction object associated with the tensor with
   * which the operation of the exponential was applied to and a shallow copy of
   * the tensor itself. This is so that during the backward pass, the derivative
   * of the resulting tensor with respect to the parent tensor can be computed
   * and passed backward such that the gradients w.r.t all subsequent parents in
   * the computational graph can be iteratively calculated via the chain rule.
   *
   * @return A new Tensor with exp applied to all elements
   *
   */
  Tensor exponential();

  /**
   * @brief Reduction sum of a Tensor along a specified dimension
   *
   * Returns a new Tensor from the summation of all values along the specified
   * dimension, reducing the rank of the tensor by one. If the parent tensor
   * requires gradient calculation as part of the construction of a
   * computational graph, then the new tensor has its gradFunction attribute set
   * to point to a GradReductionSum object. The GradReductionSum object itself
   * is instantiated with pointers to the GradFunction object associated with
   * the tensor involved in the reduction sum, as well as a shallow copy of the
   * tensor itself and the dimension along which the reduction was performed.
   * This is so that during the backward pass, the derivative of the resulting
   * tensor with respect to the parent tensor can be computed and passed
   * backward such that the gradients w.r.t all subsequent parents in the
   * computational graph can be iteratively calculated via the chain rule.
   *
   * @param dim The dimension along which to compute the sum
   * @return A new Tensor with the specified dimension summed across
   *
   * @throws std::invalid_argument If the specified dimension is less than zero
   * or greater than or equal to the rank of the tensor
   *
   */
  Tensor reductionSum(int dim);

  /**
   * @brief Elemtwise natural logarithm of a Tensor
   *
   * Returns a new Tensor from the elementwise application of the natural
   * logarithm function to all elements of this tensor. If the parent tensor
   * requires gradient calculation as part of the construction of a
   * computational graph, then the new tensor has its gradFunction attribute set
   * to point to a GradLog object. The GradLog object itself is instantiated
   * with a pointer to the GradFunction object associated with the tensor
   * involved in the elementwise logarithm, as well as a shallow copy of the
   * tensor itself. This is so that during the backward pass, the derivative of
   * the resulting tensor with respect to the parent tensor can be computed and
   * passed backward such that the gradients w.r.t all subsequent parents in the
   * computational graph can be iteratively calculated via the chain rule.
   *
   * @return A new Tensor with the natural log applied to all elements
   *
   */
  Tensor log();

  /**
   * @brief Mean of all elements of a Tensor
   *
   * Returns a one dimensional Tensor containing the arithmetic mean of all
   * elements of this tensor. If the parent tensor requires gradient calculation
   * as part of the construction of a computational graph, then the new tensor
   * has its gradFunction attribute set to point to a GradMean object. The
   * GradMean object itself is instantiated with a pointer to the GradFunction
   * objectsassociated with the tensor involved in the mean calculation, as
   * well as a shallow copy of the tensor itself. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @return A new scalar Tensor containing the mean value
   *
   */
  Tensor mean();

  Tensor sum();

  /**
   * @brief Matrix multiplication of two Tensors
   *
   * Returns a new Tensor from the matrix multiplication of two tensors. If
   * either of the parents require gradient calculation as part of the
   * construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradMultiplyMatrix object. The
   * GradMultiplyMatrix object itself is instantiated with pointers to the
   * GradFunction objects associated with each of the tensors involved in the
   * matrix multiplication, as well as shallow copies of the tensors themselves.
   * This is so that during the backward pass, the derivative of the resulting
   * tensor with respect to each of the parent tensors can be computed and
   * passed backward to each respective parent such that the gradients w.r.t
   * all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param other The right-hand tensor to multiply with
   * @return A new Tensor containing the matrix product
   *
   * @throws std::invalid_argument If either tensor is not of rank 2
   * @throws std::invalid_argument If the innermost dimensions of the two
   * tensors do not match
   *
   */
  Tensor matrixMultiply(const Tensor& other);

  /**
   * @brief Matrix multiplication between a Tensor and another Tensor having
   * been transposed
   *
   * Returns a new Tensor from the matrix multiplication of two tensors where
   * one of the tensors is transposed before multiplication. If transposeFirst
   * is true, the LHS (*this) tensor is transposed before multiplying with the
   * other tensor. If transposeFirst is false, the RHS (other) tensor is
   * transposed before being multiplied with this tensor. If either of the
   * parents require gradient calculation as part of the construction of a
   * computational graph, then the new tensor has its gradFunction attribute set
   * to point to a GradTransposeMatrix object. The GradTransposeMatrix object
   * itself is instantiated with pointers to the GradFunction objects associated
   * with each of the tensors involved in the transpose matrix multiplication,
   * as well as shallow copies of the tensors themselves and the value of the
   * transposeFirst flag. This is so that during the backward pass, the
   * derivative of the resulting tensor with respect to each of the parent
   * tensors can be computed and passed backward to each respective parent such
   * that the gradients w.r.t all subsequent parents in the computational graph
   * can be iteratively calculated via the chain rule.
   *
   * @param other The right-hand tensor to multiply with
   * @param transposeFirst If true, transpose LHS (*this) tensor before
   * multiplication; if false, transpose the RHS (other) tensor before
   * multiplication
   * @return A new Tensor containing the matrix product
   *
   * @throws std::invalid_argument If either tensor is not of rank 2
   * @throws std::invalid_argument If the innermost dimensions of the two
   * tensors do not match after the specified transposition
   *
   */
  Tensor transposeMultiply(const Tensor& other, bool transposeFirst);

  /**
   * @brief Elemtwise Rectified Linear Unit activation of a Tensor
   *
   * Returns a new Tensor from the elementwise application of the ReLU
   * function (max(0, x)) to all elements of this tensor. A backwards mask
   * tensor is also created during the forward pass, recording which elements
   * were positive and which were zeroed out. If the parent tensor requires
   * gradient calculation as part of the construction of a computational graph,
   * then the new tensor has its gradFunction attribute set to point to a
   * GradReLU object. The GradReLU object itself is instantiated with a pointer
   * to the GradFunction object associated with the tensor involved in the
   * ReLU operation, as well as the backwards mask tensor. This is so that
   * during the backward pass, the derivative of the resulting tensor with
   * respect to the parent tensor can be computed and passed backward such that
   * the gradients w.r.t all subsequent parents in the computational graph can
   * be iteratively calculated via the chain rule.
   *
   * @return A new Tensor with ReLU applied to all elements
   *
   */
  Tensor ReLU();

  /**
   * @brief Broadcasts a Tensor by inserting and repeating along a new dimension
   *
   * Returns a new Tensor created by inserting a new dimension at the specified
   * position and repeating the data along that dimension the specified number
   * of times. If the parent tensor requires gradient calculation as part of
   * the construction of a computational graph, then the new tensor has its
   * gradFunction attribute set to point to a GradBroadcast object. The
   * GradBroadcast object itself is instantiated with a pointer to the
   * GradFunction object associated with the tensor involved in the broadcast
   * operation, as well as a shallow copy of the tensor itself and the position
   * at which the new dimension was inserted. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param pos The position at which to insert the new dimension
   * @param dim The size of the new dimension (number of repetitions)
   * @return A new Tensor with the broadcast dimension added
   *
   * @throws std::invalid_argument If pos is less than zero or greater than
   * the number of dimensions
   *
   */
  Tensor broadcast(int pos, int dim);

  /**
   * @brief In-place elemtwise addition of two Tensors
   *
   * Modifies this Tensor in place by elementwise addition with another tensor.
   * A shallow copy of this tensor's pre-operation state is saved before the
   * operation is applied. If either of the tensors require gradient calculation
   * as part of the construction of a computational graph, then this tensor has
   * its gradFunction attribute updated to point to a GradAdd object. The
   * GradAdd object itself is instantiated with pointers to the GradFunction
   * objects associated with each of the tensors involved in the elementwise
   * addition, as well as shallow copies of both the pre-operation tensor and
   * the other tensor. This is so that during the backward pass, the derivative
   * of the resulting tensor with respect to each of the parent tensors can be
   * computed and passed backward to each respective parent such that the
   * gradients w.r.t all subsequent parents in the computational graph can be
   * iteratively calculated via the chain rule.
   *
   * @param other The tensor to add to this tensor
   * @return Reference to this Tensor after the operation
   *
   * @throws std::invalid_argument If the dimensions of the two tensors do not
   * match
   *
   */
  Tensor& operator+=(const Tensor& other);

  /**
   * @brief In-place elemtwise subtraction of two Tensors
   *
   * Modifies this Tensor in place by elementwise subtraction of another tensor.
   * A shallow copy of this tensor's pre-operation state is saved before the
   * operation is applied. If either of the tensors require gradient calculation
   * as part of the construction of a computational graph, then this tensor has
   * its gradFunction attribute updated to point to a GradSubtract object. The
   * GradSubtract object itself is instantiated with pointers to the
   * GradFunction objects associated with each of the tensors involved in the
   * elementwise subtraction, as well as shallow copies of both the
   * pre-operation tensor and the other tensor. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to each
   * of the parent tensors can be computed and passed backward to each
   * respective parent such that the gradients w.r.t all subsequent parents in
   * the computational graph can be iteratively calculated via the chain rule.
   *
   * @param other The tensor to subtract from this tensor
   * @return Reference to this Tensor after the operation
   *
   * @throws std::invalid_argument If the dimensions of the two tensors do not
   * match
   *
   */
  Tensor& operator-=(const Tensor& other);

  /**
   * @brief In-place elemtwise multiplication of two Tensors
   *
   * Modifies this Tensor in place by elementwise multiplication with another
   * tensor. A deep copy of this tensor's pre-operation state is saved before
   * the operation is applied. If either of the tensors require gradient
   * calculation as part of the construction of a computational graph, then this
   * tensor has its gradFunction attribute updated to point to a GradMultiply
   * object. The GradMultiply object itself is instantiated with pointers to the
   * GradFunction objects associated with each of the tensors involved in the
   * elementwise multiplication, as well as the deep copy of the pre-operation
   * tensor and the other tensor. This is so that during the backward pass, the
   * derivative of the resulting tensor with respect to each of the parent
   * tensors can be computed and passed backward to each respective parent such
   * that the gradients w.r.t all subsequent parents in the computational graph
   * can be iteratively calculated via the chain rule.
   *
   * @param other The tensor to multiply with this tensor
   * @return Reference to this Tensor after the operation
   *
   * @throws std::invalid_argument If the dimensions of the two tensors do not
   * match
   *
   */
  Tensor& operator*=(const Tensor& other);

  /**
   * @brief In-place elemtwise division of two Tensors
   *
   * Modifies this Tensor in place by elementwise division by another tensor.
   * A deep copy of this tensor's pre-operation state is saved before the
   * operation is applied. If either of the tensors require gradient calculation
   * as part of the construction of a computational graph, then this tensor has
   * its gradFunction attribute updated to point to a GradDivide object. The
   * GradDivide object itself is instantiated with pointers to the GradFunction
   * objects associated with each of the tensors involved in the elementwise
   * division, as well as the deep copy of the pre-operation tensor and the
   * other tensor. This is so that during the backward pass, the derivative of
   * the resulting tensor with respect to each of the parent tensors can be
   * computed and passed backward to each respective parent such that the
   * gradients w.r.t all subsequent parents in the computational graph can be
   * iteratively calculated via the chain rule.
   *
   * @param other The tensor to divide this tensor by
   * @return Reference to this Tensor after the operation
   *
   * @throws std::invalid_argument If the dimensions of the two tensors do not
   * match
   *
   */
  Tensor& operator/=(const Tensor& other);

  /**
   * @brief In-place addition of a scalar to all elements of this Tensor
   *
   * Modifies this Tensor in place by adding a scalar value to all elements.
   * A shallow copy of this tensor's pre-operation state is saved before the
   * operation is applied. If this tensor requires gradient calculation as part
   * of the construction of a computational graph, then this tensor has its
   * gradFunction attribute updated to point to a GradAddScalar object. The
   * GradAddScalar object itself is instantiated with a pointer to the
   * GradFunction object associated with the tensor involved in the scalar
   * addition, as well as the value of the scalar. This is so that during the
   * backward pass, the derivative of the resulting tensor with respect to the
   * parent tensor can be computed and passed backward such that the gradients
   * w.r.t all subsequent parents in the computational graph can be iteratively
   * calculated via the chain rule.
   *
   * @param scalar The scalar value to add to each element
   * @return Reference to this Tensor after the operation
   *
   */
  Tensor& operator+=(double scalar);

  /**
   * @brief In-place subtraction of a scalar from all elements of this Tensor
   *
   * Modifies this Tensor in place by subtracting a scalar value from all
   * elements. A shallow copy of this tensor's pre-operation state is saved
   * before the operation is applied. If this tensor requires gradient
   * calculation as part of the construction of a computational graph, then this
   * tensor has its gradFunction attribute updated to point to a
   * GradSubtractScalar object. The GradSubtractScalar object itself is
   * instantiated with a pointer to the GradFunction object associated with the
   * tensor involved in the scalar subtraction, as well as the value of the
   * scalar. This is so that during the backward pass, the derivative of the
   * resulting tensor with respect to the parent tensor can be computed and
   * passed backward such that the gradients w.r.t all subsequent parents in
   * the computational graph can be iteratively calculated via the chain rule.
   *
   * @param scalar The scalar value to subtract from each element
   * @return Reference to this Tensor after the operation
   *
   */
  Tensor& operator-=(double scalar);

  /**
   * @brief In-place multiplication of a scalar with all elements of this Tensor
   *
   * Modifies this Tensor in place by multiplying all elements by a scalar
   * value. A shallow copy of this tensor's pre-operation state is saved before
   * the operation is applied. If this tensor requires gradient calculation as
   * part of the construction of a computational graph, then this tensor has its
   * gradFunction attribute updated to point to a GradMultiplyScalar object. The
   * GradMultiplyScalar object itself is instantiated with a pointer to the
   * GradFunction object associated with the tensor involved in the scalar
   * multiplication, as well as the value of the scalar. This is so that during
   * the backward pass, the derivative of the resulting tensor with respect to
   * the parent tensor can be computed and passed backward such that the
   * gradients w.r.t all subsequent parents in the computational graph can be
   * iteratively calculated via the chain rule.
   *
   * @param scalar The scalar value to multiply each element by
   * @return Reference to this Tensor after the operation
   *
   */
  Tensor& operator*=(double scalar);

  /**
   * @brief In-place division of all elements of this Tensor by a scalar
   *
   * Modifies this Tensor in place by dividing all elements by a scalar value.
   * A shallow copy of this tensor's pre-operation state is saved before the
   * operation is applied. If this tensor requires gradient calculation as part
   * of the construction of a computational graph, then this tensor has its
   * gradFunction attribute updated to point to a GradDivideScalar object. The
   * GradDivideScalar object itself is instantiated with a pointer to the
   * GradFunction object associated with the tensor involved in the scalar
   * division, as well as the value of the scalar and a shared pointer to the
   * pre-operation tensor. This is so that during the backward pass, the
   * derivative of the resulting tensor with respect to the parent tensor can be
   * computed and passed backward such that the gradients w.r.t all subsequent
   * parents in the computational graph can be iteratively calculated via the
   * chain rule.
   *
   * @param scalar The scalar value to divide each element by
   * @return Reference to this Tensor after the operation
   *
   * @throws std::invalid_argument If the scalar value is zero
   *
   */
  Tensor& operator/=(double scalar);

  /**
   * @brief Checks if two tensors are elementwise equal within tolerance
   *
   * Compares if the values within each of the tensors are within a tolerance
   * of 1e-9 of being equal to each other in every element.
   *
   * @param other The tensor to compare against
   * @return True if all elements are equal within tolerance, false otherwise
   */
  bool operator==(const Tensor& other) const;

  /**
   * @brief Checks if two tensors are not elementwise equal within tolerance
   *
   * Compares if the values within each of the tensors are not within a
   * tolerance of 1e-9 of being equal to each other in every element.
   *
   * @param other The tensor to compare against
   * @return True if any elements differ beyond tolerance, false otherwise
   */
  bool operator!=(const Tensor& other) const;

  /**
   * @brief Prints the tensor values to an output stream
   *
   * @param out The output stream to print to
   */
  void print(std::ostream& out) const;

  /**
   * @brief Prints a 2D matrix representation to an output stream
   *
   * @param out The output stream to print to
   * @param spaces The number of spaces for indentation
   */
  void printMatrix(std::ostream& out, int spaces) const;

  /**
   * @brief Prints a 3D tensor representation to an output stream
   *
   * @param out The output stream to print to
   */
  void printTripleMatrix(std::ostream& out) const;

  /**
   * @brief Prints a 4D tensor representation to an output stream
   *
   * @param out The output stream to print to
   */
  void printQuadrupleMatrix(std::ostream& out) const;

  /**
   * @brief Stream insertion operator for printing tensors
   *
   * @param out The output stream
   * @param tensor The tensor to print
   * @return Reference to the output stream
   */
  friend std::ostream& operator<<(std::ostream& out, const Tensor& tensor);

  /// Sets whether this tensor currently has gradient data stored
  void setHasGrad(bool hasGrad);

  /// Returns whether this tensor currently has gradient data stored
  bool getHasGrad() const;

  /// Returns a raw pointer to the underlying contiguous data storage
  double* getData() const;

  /// Returns a raw pointer to the underlying contiguous gradient data storage
  double* getGradientData() const;

  /// Returns the strides for this tensor
  Strides getStrides() const;

  /// Returns the offset into the storage where this tensors data begins
  int getOffset() const;

  /// Returns the total number of values in the tensor
  int getNValues() const;

  /// Returns the rank (number of dimensions) of the tensor
  std::size_t getRank() const;

  /// Returns the dimensions specifying the size of each dimension
  Dims getDimensions() const;

  /**
   * @Brief Returns a shallow copy of the gradient Tensor associated with this
   * Tensor, whilst reseting the gradient Tensor of this Tensor
   *
   * Creates a shallow copy of the gradient Tensor associated with this Tensor
   * before instantiating a new gradient Tensor with zero values. Reassigns the
   * pointer owned by the GradAccumulator for this Tensor to point to the
   * new gradient Tensor such that further gradient calculations accumulate
   * gradients in the newly formed gradient object.
   *
   * @return A shallow copy of the old gradient Tensor
   *
   */
  Tensor detachGradient();

  /// Returns whether this tensor is a leaf node in the computational graph
  bool getIsLeaf() const;

  /// Returns whether this tensor requires gradient computation
  bool getRequiresGrad() const;

  /// Sets whether this tensor is a leaf node in the computational graph
  void setLeaf(bool isLeaf);

  /// Sets whether this tensor requires gradient computation
  void setRequiresGrad(bool requiresGrad);

  /// Sets the pointer to the GradFunction object for this tensor
  void setGradFunction(std::shared_ptr<GradFunction> gradFunction);

  /// Returns a shared pointer to the GradFunction object associated with this
  /// tensor
  std::shared_ptr<GradFunction> getGradFunction();
};

/**
 * @brief Addition of a scalar to all elements of a Tensor (scalar on left)
 *
 * Returns a new Tensor from the addition of a scalar value to all elements
 * of the tensor. If the tensor requires gradient calculation as part of the
 * construction of a computational graph, then the new tensor has its
 * gradFunction attribute set to point to a GradAddScalar object. The
 * GradAddScalar object itself is instantiated with a pointer to the
 * GradFunction object associated with the tensor involved in the scalar
 * addition, as well as the value of the scalar. This is so that during the
 * backward pass, the derivative of the resulting tensor with respect to the
 * parent tensor can be computed and passed backward such that the gradients
 * w.r.t all subsequent parents in the computational graph can be iteratively
 * calculated via the chain rule.
 *
 * @param scalar The scalar value to add
 * @param tensor The tensor to add the scalar to
 * @return A new Tensor with the scalar added to all elements
 *
 */
Tensor operator+(double scalar, Tensor& tensor);

/**
 * @brief Subtraction of a Tensor from a scalar (scalar on left)
 *
 * Returns a new Tensor from the subtraction of each element in the tensor
 * from the scalar value. If the tensor requires gradient calculation as part
 * of the construction of a computational graph, then the new tensor has its
 * gradFunction attribute set to point to a GradSubtractScalar object. The
 * GradSubtractScalar object itself is instantiated with a pointer to the
 * GradFunction object associated with the tensor involved in the scalar
 * subtraction, as well as the value of the scalar. This is so that during the
 * backward pass, the derivative of the resulting tensor with respect to the
 * parent tensor can be computed and passed backward such that the gradients
 * w.r.t all subsequent parents in the computational graph can be iteratively
 * calculated via the chain rule.
 *
 * @param scalar The scalar value to subtract from
 * @param tensor The tensor to subtract from the scalar
 * @return A new Tensor with (scalar - element) for each element
 *
 */
Tensor operator-(double scalar, Tensor& tensor);

/**
 * @brief Multiplication of a scalar with all elements of a Tensor (scalar on
 * left)
 *
 * Returns a new Tensor from the multiplication of a scalar value with all
 * elements of the tensor. If the tensor requires gradient calculation as part
 * of the construction of a computational graph, then the new tensor has its
 * gradFunction attribute set to point to a GradMultiplyScalar object. The
 * GradMultiplyScalar object itself is instantiated with a pointer to the
 * GradFunction object associated with the tensor involved in the scalar
 * multiplication, as well as the value of the scalar. This is so that during
 * the backward pass, the derivative of the resulting tensor with respect to
 * the parent tensor can be computed and passed backward such that the
 * gradients w.r.t all subsequent parents in the computational graph can be
 * iteratively calculated via the chain rule.
 *
 * @param scalar The scalar value to multiply by
 * @param tensor The tensor to multiply with the scalar
 * @return A new Tensor with all elements multiplied by the scalar
 *
 */
Tensor operator*(double scalar, Tensor& tensor);

/**
 * @brief Division of a scalar by each element of a Tensor (scalar on left)
 *
 * Returns a new Tensor from the division of a scalar value by each element
 * of the tensor. If the tensor requires gradient calculation as part of the
 * construction of a computational graph, then the new tensor has its
 * gradFunction attribute set to point to a GradDivideScalar object. The
 * GradDivideScalar object itself is instantiated with a pointer to the
 * GradFunction object associated with the tensor involved in the scalar
 * division, as well as the value of the scalar and a shared pointer to a
 * shallow copy of the tensor itself. This is so that during the backward
 * pass, the derivative of the resulting tensor with respect to the parent
 * tensor can be computed and passed backward such that the gradients w.r.t
 * all subsequent parents in the computational graph can be iteratively
 * calculated via the chain rule.
 *
 * @param scalar The scalar value to divide
 * @param tensor The tensor to divide the scalar by
 * @return A new Tensor with (scalar / element) for each element
 *
 */
Tensor operator/(double scalar, Tensor& tensor);
}  // namespace mattTorch
