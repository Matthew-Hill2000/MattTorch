#include <mattTorch/dataset/dataset.h>
#include <mattTorch/mattTorch.h>

#include <iostream>

void testBatchContentsUnshuffled() {
  int BATCHSIZE = 4;
  int EXAMPLESIZE = 15;
  int EPOCHS = 3;

  std::cout << "dataSet test start" << std::endl;
  mattTorch::Tensor examples({EXAMPLESIZE, 3});
  mattTorch::Tensor labels({EXAMPLESIZE, 1});

  for (int i{0}; i < examples.getDimensions()[0]; i++) {
    for (int j{0}; j < examples.getDimensions()[1]; j++) {
      examples[{i, j}] = i;
    }
    for (int j{0}; j < labels.getDimensions()[1]; j++) {
      labels[{i, j}] = i;
    }
  }

  std::cout << examples << std::endl;

  std::cout << labels << std::endl;

  mattTorch::dataset ds(BATCHSIZE, examples, labels);
  for (int i{0}; i < EPOCHS; i++) {
    for (int i{0}; i < (EXAMPLESIZE + (BATCHSIZE - 1)) / BATCHSIZE; i++) {
      std::cout << ds.getBatch()[0] << "\n";
    }
    ds.shuffle();
  }
}

int main() { testBatchContentsUnshuffled(); }
