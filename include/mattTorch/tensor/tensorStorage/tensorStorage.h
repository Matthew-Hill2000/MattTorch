#pragma once
#include <cstddef>
namespace mattTorch::tensor {

/**
 * @brief Contiguous data storage for TensorView objects.
 *
 * The TensorStorage class is responsible for holding the raw data associated
 * with a TensorView object in a contiguous block of memory. TensorStorage
 * objects are not intended to be interacted with directly, but rather through
 * the TensorView class which provides the primary interface for all tensor
 * operations. The TensorView-TensorStorage structure is designed such that
 * multiple TensorView objects can share a single TensorStorage object,
 * allowing for efficient memory usage when creating sub-tensor views or
 * shallow copies of tensors.
 *
 * @see TensorView for the primary interface by which tensor objects are
 * interacted with.
 */
class TensorStorage {
 private:
  /// Pointer to the contiguous block of memory holding the data associated
  /// with a TensorView object
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
   * The copy constructor is explicitly deleted since the TensorView class is
   * the primary interface by which we interact with tensor objects. The entire
   * point of the TensorView-TensorStorage structure is to share data and only
   * have a single TensorStorage object shared between multiple TensorView
   * objects. Copying TensorStorage objects would undermine this design.
   */
  TensorStorage(const TensorStorage& rOther) = delete;

  /**
   * @brief Destructor that deallocates the contiguous data block
   */
  ~TensorStorage();

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
