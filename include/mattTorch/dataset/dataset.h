#pragma once
#include <mattTorch/tensor/tensor/tensor.h>
namespace mattTorch {

/**
 * @brief A dataset class for managing the training data for a machine learning
 * process.
 *
 * The dataset class handles the training data for the machine learning project.
 * The features and associated labels are stored in individual Tensor objects
 * where the 0th spans accross each specific example/label in the set. A
 * vector of indices for each of the features along the example dimensions is
 * allocated during construction of the dataset. The vector of indices can be
 * shuffled, such that when the dataset is iterated over either in batches or
 * with individual features, the order during each epoch or the groupings of
 * features within the batches is not the same.
 *
 *
 * @see Tensor for the tensor class used to store the features and labels.
 */
class dataset {
 protected:
  /// Tensor of shape (numExamples x exampleSize) holding all feature data
  Tensor features;

  /// Tensor of shape (numExamples x labelSize) holding all label data
  Tensor labels;

  /// Index array used to define the ordering of features, shuffled to
  /// randomise the order in which batches are drawn
  std::vector<int> trainIndices;

  std::vector<int> valIndices;

  /// The number of features per batch
  int batchSize;

  size_t trainBatchIndex;
  size_t valBatchIndex;

 public:
  /**
   * @brief Constructs a dataset with the specified batch size and
   * features/labels
   *
   * @param batchSize The number of features per batch
   * @param features The features of the dataset
   * @param labels The labels of the dataset
   */
  dataset(int batchSize, Tensor features, Tensor labels);

  void testValSplit(double trainRation);

  /**
   * @brief Returns the next batch of features and labels within the dataset
   *
   * Constructs new Tensor objects for the batch features and batch labels
   * by copying data from the internal storage according to the current
   * shuffled index ordering. The batch starting position is calculated as
   * (batchIndex * batchSize).
   *
   * @return A vector containing two Tensor objects: the batch features
   * of shape (batchSize x exampleSize) and the batch labels of shape
   * (batchSize x labelSize)
   */
  std::vector<Tensor> getBatch(bool train = true);

  /**
   * @brief Shuffles the ordering of features using a Fisher-Yates shuffle
   *
   * Randomly permutes the internal index array so that subsequent calls to
   * getBatch will return features in a different randomised order.
   */
  void shuffle();
};
}  // namespace mattTorch
