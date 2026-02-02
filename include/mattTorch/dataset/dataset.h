#pragma once
#include <mattTorch/tensor/tensorView/tensorView.h>
namespace mattTorch {

/**
 * @brief A dataset class for loading, storing and batching training data from
 * CSV files.
 *
 * The dataset class handles the loading of labelled training data from CSV
 * files, where the first column of each row is treated as the label and the
 * remaining columns are treated as the feature values of the example. The
 * loaded data is stored internally as two TensorView objects, one for the
 * examples and one for the labels. The dataset supports shuffling of the
 * example ordering via a Fisher-Yates shuffle on an internal index array,
 * and provides batched access to the data via the getBatch method which
 * returns a batch of examples and labels as new TensorView objects copied
 * from the shuffled indices.
 *
 * @see TensorView for the tensor class used to store the examples and labels.
 */
class dataset {
 private:
  /// TensorView of shape (numExamples x exampleSize) holding all feature data
  TensorView examples;

  /// TensorView of shape (numExamples x 1) holding all label data
  TensorView labels;

  /// The total number of examples loaded from the CSV file
  int numExamples;

  /// The number of features per example (number of CSV columns minus one for
  /// the label column)
  int exampleSize;

  /// Index array used to define the ordering of examples, shuffled to
  /// randomise the order in which batches are drawn
  std::vector<int> indices;

  /// The number of examples per batch
  int batchSize;

 public:
  /**
   * @brief Constructs a dataset with the specified batch size
   *
   * @param batchSize The number of examples per batch
   */
  dataset(int batchSize);

  /**
   * @brief Returns a batch of examples and labels at the specified batch index
   *
   * Constructs new TensorView objects for the batch examples and batch labels
   * by copying data from the internal storage according to the current
   * shuffled index ordering. The batch starting position is calculated as
   * (batchIndex * batchSize).
   *
   * @param batchIndex The zero-indexed batch number to retrieve
   * @return A vector containing two TensorView objects: the batch examples
   * of shape (batchSize x exampleSize) and the batch labels of shape
   * (batchSize x 1)
   */
  std::vector<TensorView> getBatch(int batchIndex);

  /**
   * @brief Shuffles the ordering of examples using a Fisher-Yates shuffle
   *
   * Randomly permutes the internal index array so that subsequent calls to
   * getBatch will return examples in a different randomised order.
   */
  void shuffle();

  /**
   * @brief Loads labelled training data from a CSV file
   *
   * Reads the specified CSV file, skipping the first row as a header. Each
   * subsequent row is parsed as a comma-separated list of double values where
   * the first column is treated as the label and the remaining columns are
   * treated as the feature values of the example. The parsed data is stored
   * internally as two TensorView objects of shape (numExamples x exampleSize)
   * for the examples and (numExamples x 1) for the labels, and the internal
   * index array is initialised to sequential ordering.
   *
   * @param csvPath The file path to the CSV file to load
   *
   * @throws std::runtime_error If the file cannot be opened for reading
   */
  void loadData(const std::string& csvPath);

  /**
   * @brief Prints a visual representation of the first example as a 28x28
   * grid
   *
   * Intended as a debugging utility for image datasets such as MNIST. Creates
   * a 28x28 TensorView and thresholds the pixel values of the first example
   * at a value of 50, printing 1.11 for pixels above the threshold and 0.0
   * for pixels below.
   */
  void printNumber();
};
}  // namespace mattTorch
