#include <mattTorch/mattTorch.h>

#include <iostream>
#include <ostream>
#include <print>

#include "mattTorch/tensor/tensor/tensor.h"

int main(int argc, char* argv[]) {
  // At the core of the mattTorch machine learning package is the Tensor class.
  // The algorithms used within the realm of machine learning lean heavily on
  // matrix and tensor algebra. For this reason it is imperative that any
  // machine learning library worth its salt has an intuitive and heavily
  // optimised Tensor class.
  //
  // To understand the Tensor class in mattTorch we first show how to initialise
  // and manipulate Tensor objects.

  std::println("-----------------------------------------------------------");

  // The default Tensor constructor creates a single dimension tensor of length
  // 1 holding the value 0, so the print below produces [0.00].
  mattTorch::Tensor defaultTensor;

  std::cout << "default Tensor: \n" << defaultTensor << "\n";

  std::println("-----------------------------------------------------------");

  // You can create a Tensor of any arbitrary dimensions by providing the
  // constructor with the dimensions in the form of a Dims object (vector
  // wrapper)

  mattTorch::Tensor smallTensor({3, 3});

  // During the construction of a Tensor with specified dimensions, the
  // Tensor's dimensions attribute is set to be equal to the dimensions passed
  // to the constructor. A Strides object (vector wrapper) is then created and
  // set to the same size of the dimensions attribute. The Strides attribute
  // of a tensor is vital to its ability to behave as a Tensor of any
  // arbitrary dimension, and plays the central role in the ability to index a
  // Tensor along each dimension independently.
  //
  // To understand why, it is first important to understand that the values
  // stored within a tensor are stored within a TensorStorage object. Each
  // Tensor has a pointer to a TensorStorage as one of its attributes. It is
  // then within this TensorStorage object that the values are stored
  // CONTIGUOUSLY (as in sequentially in memory). The TensorStorage has two
  // attributes: a pointer to a double (the first value stored in the
  // contiguous block) named mData and the total number of values in the
  // Tensor stored in a size_t object named size.
  //
  // The Strides therefore provide the means by which to interpret this
  // contiguously stored data as a multidimensional array. The last value of
  // the Strides is always 1, since incrementing the last index (last
  // dimensions) by 1 always corresponds with the next memory address within
  // the contiguous data. The second to last value represents how large of a
  // step we must take within the contiguous data in order to increment the
  // second to last index by 1. Therefore it is equal to the size of the last
  // dimension. The third to last value of the Strides is then the size of the
  // second to last and last dimensions multiplied, since for each of N second
  // to last dimensions there are M values in the final dimension, so to
  // increment the third to last dimensions by 1 we must jump by N*M values in
  // the contiguous memory. This is known as a Row-Major layout.
  //
  // The values of the Strides attribute are filled out during initialisation
  // exactly along these lines, looping backwards through the values of the
  // dimensions attribute.
  //
  // Alongside the Strides, the Tensor also carries an offset attribute that
  // marks where its data begins within the contiguous storage. For a Tensor
  // constructed from scratch the offset is zero, and so it does not show up
  // in any of the indexing arithmetic discussed above. The role of the
  // offset becomes important once we start producing sub-tensor views that
  // share storage with a parent Tensor, and is returned to further down in
  // this tutorial.
  //
  // The TensorStorage object pointed to by the storage attribute is then
  // initialised with the number of values calculated as the product of the
  // dimensions attribute. The allocation of memory for the values is then
  // performed using posix_memalign. The posix_memalign function takes a
  // pointer to our mData pointer, the number of bytes we need (so number of
  // values x sizeof(double)) and then allows us to specify an integer that
  // dictates the alignment of the memory allocated. mattTorch specifies an
  // alignment of 64, meaning the allocated memory starts at a memory address
  // that is a multiple of 64. This is important later for efficiently loading
  // aligned cache lines into registers in order to leverage SIMD operations.
  // The memset function is then used to initialise all of the values to
  // zero.

  // Assigning a scalar value to the Tensor results in all values taking on
  // the scalar value. This operation doesn't require the use of the Strides,
  // since of course every element receives the same value anyway. The
  // std::fill function is called on the TensorStorage mData attribute to set
  // all of the values

  smallTensor = 2.0;
  std::cout << "smallTensor: \n" << smallTensor << '\n';

  std::println("-----------------------------------------------------------");

  mattTorch::Tensor largeTensor({3, 3, 3});

  // Alternatively you can index the specific elements of a Tensor with a Dims
  // object of integer indices. The following code sets the value of each
  // element of the tensor equal to the sum of its indices.
  //
  // When a Tensor is indexed across all of its dimensions in this way, the
  // Tensor class uses a helper function calculateIndex to convert the vector
  // of indices into the equivalent index within the contiguous data storage.
  // calculateIndex multiplies each of the provided indices with the
  // associated value from strides, summing each multiplication to get the
  // contiguous index. The [] accessor then returns the value in storage by
  // reference, allowing it to be modified.
  //
  // The alternative methods setValue and getValue also exist, and function in
  // exactly the same way. Alternatively, the setValueDirect and getValueDirect
  // methods allow you to index, access and modify data within the Tensor
  // using a linear index across its own view of the contiguous storage. The
  // offset of the Tensor is added to this index internally, so for a Tensor
  // with offset zero (as below) it lines up exactly with the index into the
  // underlying TensorStorage, and for a sub-tensor view it indexes from the
  // start of that view.
  //

  for (int i{0}; i < largeTensor.getDimensions()[0]; i++) {
    for (int j{0}; j < largeTensor.getDimensions()[1]; j++) {
      for (int k{0}; k < largeTensor.getDimensions()[2]; k++) {
        largeTensor[{i, j, k}] = i + j + k;
      }
    }
  }

  std::cout << "largeTensor: \n" << largeTensor << '\n';

  largeTensor.setValueDirect(11, 999);
  std::cout << "largeTensor: \n" << largeTensor << '\n';
  std::cout << "largeTensor.getValueDirect(11): "
            << largeTensor.getValueDirect(11) << '\n';

  // The getData method returns the pointer to the start of the contiguous
  // data storage, held by TensorStorage
  //

  size_t size = largeTensor.getNValues();
  double* data = largeTensor.getData();

  std::println("largeTensor contiguous storage: \n");

  for (auto i{0}; i < size; i++) {
    std::cout << *(data + i) << " ";
  }
  std::cout << "\n";

  std::println("-----------------------------------------------------------");

  // The Tensor class has nice formatting defined up to four dimensions for
  // printing the tensor to the terminal

  largeTensor = mattTorch::Tensor({3, 3, 3, 3});
  for (int i{0}; i < largeTensor.getDimensions()[0]; i++) {
    for (int j{0}; j < largeTensor.getDimensions()[1]; j++) {
      for (int k{0}; k < largeTensor.getDimensions()[2]; k++) {
        for (int l{0}; l < largeTensor.getDimensions()[3]; l++) {
          largeTensor[{i, j, k, l}] = i + j + k + l;
        }
      }
    }
  }

  std::cout << "largeTensor: \n" << largeTensor << '\n';

  std::println("-----------------------------------------------------------");

  // Assigning one tensor to another performs a shallow copy where each of the
  // original Tensor's attributes are copied into the new Tensor. The copy
  // constructor for the Tensor is set to default, meaning all attributes are
  // simply copied directly. Since the TensorStorage object is accessed via a
  // pointer from within the Tensor class, both the original and resulting
  // Tensor objects end up sharing the same underlying data, since both point
  // to the same underlying data. If either Tensor makes alterations to their
  // values, the same changes are reflected in the other tensor.

  mattTorch::Tensor shallowCopyTensor = smallTensor;
  std::cout << "shallowCopyTensor: \n" << shallowCopyTensor << '\n';

  smallTensor[{0, 0}] = 999;
  std::cout << "shallowCopyTensor: \n" << shallowCopyTensor << '\n';

  std::cout << "smallTensor: \n" << smallTensor << '\n';

  std::println("-----------------------------------------------------------");

  // Using the deep copy method lets us create a copy where the new tensor
  // possesses its own independent copy of the data. The deepCopy method
  // constructs a new tensor, and then directly copies each of the values in
  // the contiguous data storage from the original tensor into the new tensor.

  mattTorch::Tensor deepCopyTensor = smallTensor.deepCopy();
  std::cout << "deepCopyTensor: \n" << deepCopyTensor << '\n';

  smallTensor[{0, 0}] = 666;

  std::cout << "deepCopyTensor: \n" << deepCopyTensor << '\n';
  std::cout << "smallTensor: \n" << smallTensor << '\n';

  // You can also index a tensor with a single integer. This returns a view
  // along the first dimension at the given index, with that dimension dropped
  // from the resulting Tensor. So for a tensor with dimensions {3,3,3},
  // Tensor[1] returns the slice at index 1 of the first dimension, leaving a
  // Tensor with dimensions {3,3}. For a tensor with dimensions {4,4},
  // Tensor[2] returns the third row. To reduce the overhead associated with
  // copying data from large tensors, the resulting Tensor shares its
  // underlying storage with the original Tensor, so both Tensors have
  // pointers to the same TensorStorage. For this reason, the pointer held by
  // the Tensor class to its associated TensorStorage is specifically a SHARED
  // POINTER, such that a TensorStorage object remains alive so long as any
  // one Tensor continues to point to it.
  //
  // When indexing a Tensor with an integer, it is always the first dimension
  // that we are indexing and reducing. The resulting tensor therefore has the
  // same Dims and Strides as the original, except for having their first
  // values removed. This is also where the Tensor class' offset attribute
  // mentioned earlier finally comes in to play, since the indexed Tensor
  // shares the entire original TensorStorage whilst only requiring a subset.
  //
  // The first value of the original's strides tells us how far we must jump
  // in contiguous data when we increment the first dimension by 1, so for a
  // Tensor indexed with integer INDEX, we know that its values will start at
  // INDEX*strides[0]. We set this value to the indexed Tensor's offset, and
  // it plays a role in all of the indexing methods and calculations so far
  // discussed, appropriately adding the offset when required.

  std::println("-----------------------------------------------------------");

  std::cout << "largeTensor: \n" << largeTensor << '\n';

  auto indexedLargeTensor = largeTensor[1];
  std::cout << "largeTensor[1]: \n" << indexedLargeTensor << '\n';

  indexedLargeTensor[{0, 1, 2}] = 9696.0;

  std::cout << "largeTensor: \n" << largeTensor << '\n';
  std::cout << "largeTensor[1]: \n" << indexedLargeTensor << '\n';

  return 0;
}
