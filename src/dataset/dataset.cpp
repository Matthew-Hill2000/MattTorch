#include <mattTorch/dataset/dataset.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

namespace mattTorch {

dataset::dataset(int batchSize, Tensor features, Tensor labels)
    : features{std::move(features)},
      labels{std::move(labels)},
      batchSize(batchSize),
      trainBatchIndex{0},
      valBatchIndex{0} {
  trainIndices.resize(this->features.getDimensions()[0]);
  std::iota(trainIndices.begin(), trainIndices.end(), 0);
}

void dataset::shuffle() {
  if (trainIndices.size() < 2) {
    return;
  }

  std::random_device rd;
  std::mt19937 gen(rd());

  for (size_t i = trainIndices.size() - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t j = dist(gen);
    std::swap(trainIndices[i], trainIndices[j]);
  }
}

void dataset::testValSplit(double trainRatio) {
  assert(trainRatio >= 0 && trainRatio < 1.0);

  this->valIndices =
      std::vector<int>(trainIndices.begin() + trainIndices.size() * trainRatio,
                       trainIndices.end());

  this->trainIndices =
      std::vector<int>(trainIndices.begin(),
                       trainIndices.begin() + trainIndices.size() * trainRatio);
}

std::vector<Tensor> dataset::getBatch(bool train) {
  const std::vector<int>& indices = train ? trainIndices : valIndices;
  size_t* batchIndex = train ? &trainBatchIndex : &valBatchIndex;

  assert(this->features.getDimensions()[0] > 0);
  assert(batchSize > 0);

  int batchStart = *batchIndex * batchSize;
  int batchEnd =
      std::min(batchStart + batchSize, static_cast<int>(indices.size()));

  assert(std::cmp_less_equal(batchEnd, indices.size()) &&
         "Batch index out of range");

  Dims batchExamplesDims = this->features.getDimensions();
  batchExamplesDims[0] = batchEnd - batchStart;
  Tensor batchExamples(batchExamplesDims);

  Dims batchLabelsDims = this->labels.getDimensions();
  batchLabelsDims[0] = batchEnd - batchStart;
  Tensor batchLabels(batchLabelsDims);

  double* featuresData = features.getData();
  double* labelsData = labels.getData();

  double* batchExamplesData = batchExamples.getData();
  double* batchLabelsData = batchLabels.getData();

  for (int i = 0; i < batchEnd - batchStart; i++) {
    int datasetIdx = indices[batchStart + i];

    std::memcpy(batchExamplesData + i * batchExamples.getStrides()[0],
                featuresData + this->features.getStrides()[0] * datasetIdx,
                batchExamples.getStrides()[0] * sizeof(double));

    std::memcpy(batchLabelsData + i * batchLabels.getStrides()[0],
                labelsData + this->labels.getStrides()[0] * datasetIdx,
                batchLabels.getStrides()[0] * sizeof(double));
  }

  (*batchIndex)++;
  *batchIndex = *batchIndex % ((indices.size() + (batchSize - 1)) / batchSize);
  return {batchExamples, batchLabels};
}

}  // namespace mattTorch
