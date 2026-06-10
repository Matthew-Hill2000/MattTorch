#pragma once

#include <initializer_list>
#include <numeric>
#include <vector>
namespace mattTorch {

/**
 * @brief The Strides object is a vector of integers used to store how many
 * elements one must step through the contiguous memory of a Tensor in order to
 * increment a specific multi-dimensional index by one.
 *
 * The storage attribute of a Tensor is a shared pointer to a TensorStorage
 * object which stores the Tensors values in a single flat, contiguous buffer of
 * memory. The Strides are a necessary component of it being possible to
 * interpret this flat memory as a multi-dimensional tensor. For each element of
 * the dimensions vector, the same element within the Strides vector tells us
 * how far we must jump in memory if we were to increment that dimension by a
 * single unit.
 *
 */
struct Strides {
  // A vector containing the Stride values.
  std::vector<int> values;

  /**
   * @brief The default constructor for Strides, initialising an empty vector.
   *
   * The default constructor calls the default constructor of the std::vector
   * that the Strides objects values are contained within. This results in
   * the values vector and therefore the Strides being initialised as an empty,
   * but usable vector.
   */
  Strides() = default;

  /*
   * @brief A Constructor for the Strides object that initialises its values
   * with the provided vector of integers
   *
   * An initaliser list is used in this constructor to directly initialise the
   * values of the Strides underlying vector of values during construction.
   *
   * @param init An initialiser list of integer values representing the stride
   * along each dimensions
   */
  Strides(std::initializer_list<int> init) : values{init} {};

  /**
   * @brief A constructor for the Strides object that initialises its values
   * from an iterator range.
   *
   * The values of the Strides underlying vector are initialised from the range
   * defined by the first and last iterators.
   *
   * @param first An iterator pointing to the first value to copy.
   * @param last An iterator pointing one past the final value to copy.
   */
  template <typename It>
  Strides(It first, It last) : values(first, last) {}

  /*
   * @brief resize the underlying vector of strides
   *
   * Calls the resize method on the underlying std::vector<int>
   * of strides values.
   *
   * @param n The size with which the underlying vector will be resized to
   */
  void resize(size_t n) { values.resize(n); }

  /*
   * @brief Returns the current size of the Strides vector
   *
   * calles the .size() method of the underlying std::vector<int> of
   * strides.
   *
   * @return A size_t object representing the size of the Strides vector
   *
   */
  size_t size() const { return values.size(); }

  /*
   * @brief return a reference the value of the stride for the indexed dimension
   *
   * Calls the .at() method on the underlying std::vector<int> of
   * stride values to return the stride value at that dimensions.
   * The returned value will represent how much one must step through
   * the contiguous storage of the Tensors values in order to increment
   * the index dimensions by one.
   *
   * @param index the dimension for which the stride shall be returned for
   * @return A reference to the stride value for the index dimension
   *
   */
  int& operator[](size_t index) { return values.at(index); }

  /*
   * @brief return a const reference to the value of the stride for the indexed
   * dimension
   *
   * Calls the .at() method on the underlying std::vector<int> of
   * stride values to return the stride value at that dimension.
   * The returned value will represent how much one must step through
   * the contiguous storage of the Tensors values in order to increment
   * the index dimensions by one.
   *
   * @param index the dimension for which the stride shall be returned for
   * @return A const reference to the stride value for the index dimension
   *
   */
  const int& operator[](size_t index) const { return values.at(index); }

  /*
   * @brief Returns an iterator pointing to the start of the Strides vector
   *
   * calls the .begin() method of the underlying std::vector<int> of values
   * to return an iterator pointing to the start of the underling vector
   *
   * @return An iterator pointing to the start of the Strides vector
   */
  auto begin() { return values.begin(); }

  /*
   * @brief Returns an iterator pointing to the end of the Strides vector
   *
   * calls the .end() method of the underlying std::vector<int> of values
   * to return an iterator pointing to the end of the underling vector
   *
   * @return An iterator pointing to the end of the Strides vector
   */
  auto end() { return values.end(); }

  /*
   * @brief Returns an iterator pointing to the start of the Strides vector
   *
   * calls the .begin() method of the underlying std::vector<int> of values
   * to return an iterator pointing to the start of the underling vector
   *
   * @return An iterator pointing to the start of the Strides vector
   */
  auto begin() const { return values.begin(); }

  /*
   * @brief Returns an iterator pointing to the end of the Strides vector
   *
   * calls the .end() method of the underlying std::vector<int> of values
   * to return an iterator pointing to the end of the underling vector
   *
   * @return An iterator pointing to the end of the Strides vector
   */
  auto end() const { return values.end(); }

  /*
   * @brief Check if one Strides object contains the same values as another
   * Strides object
   *
   * Uses the == operator of the underlying std::vector<int> object to check if
   * both Strides objects have equivalent values in their underlying vectors.
   *
   * @return A boolean value representing whether or not the two Stride objects
   * are equivalent
   */
  bool operator==(const Strides& other) const { return values == other.values; }

  /*
   * @brief Check if one Strides object contains different values as another
   * Strides object
   *
   * Uses the != operator of the underlying std::vector<int> object to check if
   * both Strides objects have differing values in their underlying vectors.
   *
   * @return A boolean value representing whether or not the two Stride objects
   * are different
   */
  bool operator!=(const Strides& other) const { return !(*this == other); }

  /*
   * @brief append a value to the end of the Strides vector
   *
   * Calls the push_back() method of the underlying std::vector<int> of values
   * to append a value to the end of the Strides vector.
   *
   * @param value The value to append to the end of the vector
   */
  void push_back(int value) { values.push_back(value); }

  /*
   * @brief Check if the Strides vector is empty
   *
   * Calls the .empty() method of the underlying std::vector<int> of values to
   * check if the values vector is empty
   *
   * @return A boolean value representing if the Strides vector is empty
   */
  bool empty() const { return values.empty(); }

  /*
   * @brief Insert a value into the Strides vector at a specific location
   *
   * Uses the .insert() method of the underlying std::vector<int> of values
   * to insert a new value in the Strides vector at a position given by an
   * iterator.
   *
   * @param iter An iterator pointing to the location at which the new value
   * will be inserted into the Strides vector
   * @return The Strides vector iteself, with the newly inserted vector
   */
  auto insert(auto iter, int v) { return values.insert(iter, v); }
};

/**
 * @brief The Dims object is a vector of integers used to store the size of each
 * dimension of a Tensor.
 *
 * The storage attribute of a Tensor is a shared pointer to a TensorStorage
 * object which stores the Tensors values in a single flat, contiguous buffer of
 * memory. The Dims are a necessary component of it being possible to interpret
 * this flat memory as a multi-dimensional tensor. For each dimension of the
 * Tensor, theelement within the Dims vector tells us the size of that
 * dimension.
 *
 */
struct Dims {
  /// A vector containing the dimension sizes.
  std::vector<int> values;

  /**
   * @brief The default constructor for Dims, initialising an empty vector.
   *
   * The default constructor calls the default constructor of the std::vector
   * that the Dims objects values are contained within. This results in the
   * values vector and therefore the Dims being initialised as an empty, but
   * usable vector.
   */
  Dims() = default;

  /**
   * @brief A constructor for the Dims object that initialises its values with
   * the provided vector of integers.
   *
   * An initialiser list is used in this constructor to directly initialise the
   * values of the Dims underlying vector of values during construction.
   *
   * @param init An initialiser list of integer values representing the size of
   * each dimension.
   */
  Dims(std::initializer_list<int> init) : values{init} {};

  /**
   * @brief A constructor for the Dims object that initialises its values from
   * an iterator range.
   *
   * The values of the Dims underlying vector are initialised from the range
   * defined by the first and last iterators.
   *
   * @param first An iterator pointing to the first value to copy.
   * @param last An iterator pointing one past the final value to copy.
   */
  template <typename It>
  Dims(It first, It last) : values(first, last) {};

  using value_type = int;

  /**
   * @brief Resize the underlying vector of dimensions.
   *
   * Calls the resize method on the underlying std::vector<int> of dimension
   * values.
   *
   * @param n The size with which the underlying vector will be resized to.
   */
  void resize(size_t n) { values.resize(n); }

  /**
   * @brief Returns the current size of the Dims vector.
   *
   * Calls the .size() method of the underlying std::vector<int> of dimensions.
   *
   * @return A size_t object representing the size of the Dims vector.
   */
  size_t size() const { return values.size(); }

  /**
   * @brief Return a reference to the size of the indexed dimension.
   *
   * Calls the .at() method on the underlying std::vector<int> of dimension
   * values to return the size of the indexed dimension.
   *
   * @param index The dimension for which the size shall be returned.
   * @return A reference to the size of the indexed dimension.
   */
  int& operator[](size_t index) { return values.at(index); }

  /**
   * @brief Return a const reference to the size of the indexed dimension.
   *
   * Calls the .at() method on the underlying std::vector<int> of dimension
   * values to return the size of the indexed dimension.
   *
   * @param index The dimension for which the size shall be returned.
   * @return A const reference to the size of the indexed dimension.
   */
  const int& operator[](size_t index) const { return values.at(index); }

  /**
   * @brief Returns an iterator pointing to the start of the Dims vector.
   *
   * Calls the .begin() method of the underlying std::vector<int> of values to
   * return an iterator pointing to the start of the underlying vector.
   *
   * @return An iterator pointing to the start of the Dims vector.
   */
  auto begin() { return values.begin(); }

  /**
   * @brief Returns an iterator pointing to the end of the Dims vector.
   *
   * Calls the .end() method of the underlying std::vector<int> of values to
   * return an iterator pointing to the end of the underlying vector.
   *
   * @return An iterator pointing to the end of the Dims vector.
   */
  auto end() { return values.end(); }

  /**
   * @brief Returns an iterator pointing to the start of the Dims vector.
   *
   * Calls the .begin() method of the underlying std::vector<int> of values to
   * return an iterator pointing to the start of the underlying vector.
   *
   * @return An iterator pointing to the start of the Dims vector.
   */
  auto begin() const { return values.begin(); }

  /**
   * @brief Returns an iterator pointing to the end of the Dims vector.
   *
   * Calls the .end() method of the underlying std::vector<int> of values to
   * return an iterator pointing to the end of the underlying vector.
   *
   * @return An iterator pointing to the end of the Dims vector.
   */
  auto end() const { return values.end(); }

  /**
   * @brief Check if one Dims object contains the same values as another Dims
   * object.
   *
   * Uses the == operator of the underlying std::vector<int> object to check if
   * both Dims objects have equivalent values in their underlying vectors.
   *
   * @param other The Dims object to compare with.
   * @return A boolean value representing whether or not the two Dims objects
   * are equivalent.
   */
  bool operator==(const Dims& other) const { return values == other.values; }

  /**
   * @brief Check if one Dims object contains different values from another Dims
   * object.
   *
   * Uses the != operator of the underlying std::vector<int> object to check if
   * both Dims objects have differing values in their underlying vectors.
   *
   * @param other The Dims object to compare with.
   * @return A boolean value representing whether or not the two Dims objects
   * are different.
   */
  bool operator!=(const Dims& other) const { return !(*this == other); }

  /**
   * @brief Append a value to the end of the Dims vector.
   *
   * Calls the push_back() method of the underlying std::vector<int> of values
   * to append a value to the end of the Dims vector.
   *
   * @param value The value to append to the end of the vector.
   */
  void push_back(int value) { values.push_back(value); }

  /**
   * @brief Check if the Dims vector is empty.
   *
   * Calls the .empty() method of the underlying std::vector<int> of values to
   * check if the values vector is empty.
   *
   * @return A boolean value representing if the Dims vector is empty.
   */
  bool empty() const { return values.empty(); }

  /**
   * @brief Insert a value into the Dims vector at a specific location.
   *
   * Uses the .insert() method of the underlying std::vector<int> of values to
   * insert a new value in the Dims vector at a position given by an iterator.
   *
   * @param iter An iterator pointing to the location at which the new value
   * will be inserted into the Dims vector.
   * @param v The value to insert into the Dims vector.
   * @return An iterator pointing to the inserted value.
   */
  auto insert(auto iter, int v) { return values.insert(iter, v); }

  /**
   * @brief Return the product of all dimension sizes.
   *
   * Calls std::accumulate on the underlying std::vector<int> of dimension
   * values to calculate the total number of values represented by the Dims
   * object.
   *
   * @return An integer representing the product of all dimension sizes.
   */
  int product() const {
    return std::accumulate(values.cbegin(), values.cend(), 1,
                           std::multiplies<int>{});
  }
};

/*
 * @brief The metadata associated with interpreting the flat, contiguous memory
 * of a Tensor as a multidimensional array.
 */
struct TensorData {
  /**
   * @brief The default constructor for TensorData
   *
   * The default constuctor calls the default constructors of Dims and Strides
   * to initalize dimensions and strides, and default initaliases the offset
   * attribute
   */
  TensorData() = default;

  /*
   * @brief A Constructor for the TensorData object that initialises its values
   * with the provided parameters for each attribute
   *
   * This constructor takes a Dims, Strides and int parameter that are used to
   * initialize the dimensions, strides and offset parameters using a member
   * initializer list, in which the dimensions and strides are moved using
   * std::move into their associated attributes.
   *
   * @param dimensions A Dims vector representing the size of each dimension
   * @param strides A Strides vector that represents how many elements to step
   * by in the contiguous memory in order to increment a specific dimension by
   * one
   * @param offset the location within contiguous memory where the data for the
   * associated sub tensor begins
   */
  TensorData(Dims dimensions, Strides strides, int offset)
      : dimensions(std::move(dimensions)),
        strides(std::move(strides)),
        offset(offset) {}

  // The Dims object is a vector of integers used to store the size of each
  // dimension of a Tensor.
  Dims dimensions;

  // The Strides object is a vector of integers used to store how many
  // elements one must step through the contiguous memory of a Tensor in order
  // to increment a specific multi-dimensional index by one.
  Strides strides;

  // The offset is used by sub tensors to define where in the underlying
  // contiguous storage of values do the values for this sub tensors begin.
  int offset{};
};

/*
 * @brief The metadata associated with the construction and subsequent traversal
 * of the computational graph that is utilised in the computation of gradients.
 */
struct GradData {
  // A boolean value that describes whether this tensor sits at the end of a
  // computational graph
  bool isLeaf;

  // A boolean that states whether this Tensor shall require gradients being
  // calculated with respect to itself in the backward pass of any associated
  // computational graphs.
  bool requiresGrad;

  // A boolean value that states whether the associated Tensor has any gradient
  // values so far computed and stored within its .gradient attribute
  bool hasGrad;
};

}  // namespace mattTorch
