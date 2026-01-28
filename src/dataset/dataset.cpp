#include <mattTorch/dataset/dataset.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <algorithm>
#include <cassert>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace mattTorch {

dataset::dataset(int batchSize)
    : numExamples(0), exampleSize(0), batchSize(batchSize) {}

void dataset::loadData(const std::string& csvPath) {
  std::ifstream file(csvPath);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to open CSV file: " + csvPath);
  }

  std::vector<double> examplesBuffer;
  std::vector<double> labelsBuffer;

  std::string line;
  int detectedColumnCount = -1;
  int rowCount = 0;

  // Skip header
  std::getline(file, line);

  while (std::getline(file, line)) {
    if (line.empty()) continue;

    std::stringstream ss(line);
    std::string cell;
    int columnCount = 0;

    while (std::getline(ss, cell, ',')) {
      double value = std::stod(cell);

      if (columnCount == 0) {
        labelsBuffer.push_back(value);
      } else {
        examplesBuffer.push_back(value);
      }

      columnCount++;
    }

    if (detectedColumnCount == -1) {
      detectedColumnCount = columnCount;
    } else {
      assert(columnCount == detectedColumnCount &&
             "Inconsistent number of columns in CSV");
    }

    rowCount++;
  }

  assert(rowCount > 0);
  assert(detectedColumnCount >= 2);

  numExamples = rowCount;
  exampleSize = detectedColumnCount - 1;

  assert(static_cast<int>(labelsBuffer.size()) == numExamples);
  assert(static_cast<int>(examplesBuffer.size()) == numExamples * exampleSize);

  examples = TensorView({numExamples, exampleSize});
  labels = TensorView({numExamples, 1});

  std::memcpy(examples.getData(), examplesBuffer.data(),
              examplesBuffer.size() * sizeof(double));

  std::memcpy(labels.getData(), labelsBuffer.data(),
              labelsBuffer.size() * sizeof(double));

  indices.resize(numExamples);
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

std::vector<TensorView> dataset::getBatch(int batchIndex) {
  assert(numExamples > 0);
  assert(batchSize > 0);

  int batchStart = batchIndex * batchSize;
  int batchEnd = batchStart + batchSize;

  assert(batchEnd <= numExamples && "Batch index out of range");

  TensorView batchExamples({batchSize, exampleSize});
  TensorView batchLabels({batchSize, 1});

  double* dstExamples = batchExamples.getData();
  double* dstLabels = batchLabels.getData();

  double* srcExamples = examples.getData();
  double* srcLabels = labels.getData();

  for (int j = 0; j < batchSize; ++j) {
    int datasetIdx = indices[batchStart + j];

    // Copy example row
    std::memcpy(dstExamples + j * exampleSize,
                srcExamples + datasetIdx * exampleSize,
                exampleSize * sizeof(double));

    // Copy label
    dstLabels[j] = srcLabels[datasetIdx];
  }

  return {batchExamples, batchLabels};
}

void dataset::printNumber() {
  TensorView number({28, 28});
  double* numberData = number.getData();
  double* thisData = examples.getData();

  for (int i{0}; i < number.getNValues(); i++) {
    if (thisData[i] > 50) {
      numberData[i] = 1.11;
    } else {
      numberData[i] = 0.0;
    }
  }

  std::cout << number << std::endl;
}

}  // namespace mattTorch
