#include <mattTorch/mattTorch.h>

#include <random>

#include "mattTorch/dataset/dataset.h"

namespace mattTorch {

class SyntheticRegressionDataset : public dataset {
 public:
  SyntheticRegressionDataset(const Tensor& weight, Tensor& bias, double noise,
                             int numTrain, int batchSize)
      : dataset(batchSize, Tensor({numTrain, weight.getNValues()}),
                Tensor({numTrain, 1})) {
    const int numFeatures = weight.getNValues();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> d(0.0, 1.0);

    for (int i = 0; i < numTrain * numFeatures; ++i) {
      this->examples.setValueDirect(i, d(gen));
    }

    Tensor error({numTrain, 1});
    for (int i = 0; i < numTrain; ++i) {
      error.setValueDirect(i, d(gen) * noise);
    }

    this->labels = this->examples.matrixMultiply(weight) +
                   bias.broadcast(0, numTrain) + error;
  }
};
}  // namespace mattTorch
