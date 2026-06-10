#pragma once
#include <cstddef>
namespace mattTorch::tensor {

/**
 * @brief Contiguous data storage for Tensor objects.
 *
 * The TensorStorage class is responsible for holding the raw data associated
 * with a Tensor object in a contiguous block of memory. TensorStorage
 * objects are not intended to be interacted with directly, but rather through
 * the Tensor class which provides the primary interface for all tensor
 * operations. The Tensor-TensorStorage structure is designed such that
 * multiple Tensor objects can share a single TensorStorage object,
 * allowing for efficient memory usage when creating sub-tensor views or
 * shallow copies of tensors.
 *
 * @see Tensor for the primary interface by which tensor objects are
 * interacted with.
 */
class TensorStorage {
 private:
  /// Pointer to the contiguous block of memory holding the data associated
  /// with a Tensor object
  double* mData;

  /// The number of elements stored in the contiguous data block
  size_t size;

 public:
  /**
   * @brief Creates a TensorStorage object with the capacity for the specified
   * number of elements
   *
   * Allocates a contiguous block of memory with the capacity for the specified
   * number of elements.
   *
   * @param size The number of elements to allocate storage for
   */
  TensorStorage(size_t size);

  /**
   * @brief Creates a TensorStorage object by assuming ownership of a
   * pre-existing data pointer
   *
   * Uses move semantics to assume ownership of a premade array of double
   * values, avoiding any unnecessary copying of data.
   *
   * @param rValues Pointer to the pre-existing data to assume ownership of
   */
  TensorStorage(double* rValues);

  /**
   * @brief Copy constructor is deleted
   *
   * The copy constructor is explicitly deleted since the Tensor class is
   * the primary interface by which we interact with tensor objects. The entire
   * point of the Tensor-TensorStorage structure is to share data and only
   * have a single TensorStorage object shared between multiple Tensor
   * objects. Copying TensorStorage objects would undermine this design.
   */
  TensorStorage(const TensorStorage& rOther) = delete;

  /**
   * @brief Destructor that deallocates the contiguous data block
   */
  ~TensorStorage();

  /**
   * @brief Copy assignment operator
   *
   * Performs the default member-wise assignment of the TensorStorage object.
   * This copies the data pointer and size from the other TensorStorage object.
   *
   * @param other The TensorStorage object to copy assign from
   * @return Reference to this TensorStorage after assignment
   */
  TensorStorage& operator=(TensorStorage const& other) = default;

  /**
   * @brief Move assignment operator
   *
   * Performs the default member-wise move assignment of the TensorStorage
   * object. Since the data pointer is a raw pointer, the pointer value itself
   * is assigned from the other TensorStorage object.
   *
   * @param other The TensorStorage object to move assign from
   * @return Reference to this TensorStorage after assignment
   */
  TensorStorage& operator=(TensorStorage&& other) = default;

  /**
   * @brief Move constructor
   *
   * Performs the default member-wise move construction of the TensorStorage
   * object. Since the data pointer is a raw pointer, the pointer value itself
   * is copied from the other TensorStorage object.
   *
   * @param other The TensorStorage object to move construct from
   */
  TensorStorage(TensorStorage&& other) = default;

  /**
   * @brief Returns a reference to the element at the specified index within
   * the contiguous data block
   *
   * @param index The index of the element to access
   * @return A reference to the element at the specified index
   */
  double& at(int index);

  /**
   * @brief Returns a const reference to the element at the specified index
   * within the contiguous data block
   *
   * @param index The index of the element to access
   * @return A const reference to the element at the specified index
   */
  const double& at(int index) const;

  /// Returns the number of elements stored in the contiguous data block
  size_t getSize() const;

  /**
   * @brief Returns a pointer to the contiguous data block
   *
   * @return A pointer to the underlying data
   */
  double* getData();

  /**
   * @brief Returns a const pointer to the contiguous data block
   *
   * @return A const pointer to the underlying data
   */
  const double* getData() const;

  /**
   * @brief Sets all elements in the contiguous data block to the specified
   * value
   *
   * @param value The value to set all elements to
   */
  void setAllValues(double value);
};
}  // namespace mattTorch::tensor
