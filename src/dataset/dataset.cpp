#include <mattTorch/dataset/dataset.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <vector>

namespace mattTorch {

dataset::dataset(int batchSize, Tensor examples, Tensor labels)
    : examples{std::move(examples)},
      labels{std::move(labels)},
      batchSize(batchSize),
      batchIndex{0} {
  indices.resize(this->examples.getDimensions()[0]);
  std::iota(indices.begin(), indices.end(), 0);
}

void dataset::shuffle() {
  if (indices.size() < 2) return;

  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t i = indices.size() - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t j = dist(gen);
    std::swap(indices[i], indices[j]);
  }
}

std::vector<Tensor> dataset::getBatch() {
  assert(this->examples.getDimensions()[0] > 0);
  assert(batchSize > 0);

  int batchStart = batchIndex * batchSize;
  int batchEnd =
      std::min(batchStart + batchSize, this->examples.getDimensions()[0]);

  assert(batchEnd <= this->examples.getDimensions()[0] &&
         "Batch index out of range");

  Dims batchExamplesDims = this->examples.getDimensions();
  batchExamplesDims[0] = batchEnd - batchStart;
  Tensor batchExamples(batchExamplesDims);

  Dims batchLabelsDims = this->labels.getDimensions();
  batchLabelsDims[0] = batchEnd - batchStart;
  Tensor batchLabels(batchLabelsDims);

  double* examplesData = examples.getData();
  double* labelsData = labels.getData();

  double* batchExamplesData = batchExamples.getData();
  double* batchLabelsData = batchLabels.getData();

  for (int i = 0; i < batchEnd - batchStart; i++) {
    int datasetIdx = indices[batchStart + i];

    std::memcpy(batchExamplesData + i * batchExamples.getStrides()[0],
                examplesData + this->examples.getStrides()[0] * datasetIdx,
                batchExamples.getStrides()[0] * sizeof(double));

    std::memcpy(batchLabelsData + i * batchLabels.getStrides()[0],
                labelsData + this->labels.getStrides()[0] * datasetIdx,
                batchLabels.getStrides()[0] * sizeof(double));
  }

  batchIndex++;
  batchIndex = batchIndex %
               ((examples.getDimensions()[0] + (batchSize - 1)) / batchSize);

  return {batchExamples, batchLabels};
}

}  // namespace mattTorch
