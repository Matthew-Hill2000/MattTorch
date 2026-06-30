#include <mattTorch/mattTorch.h>

#include <random>

#include "mattTorch/dataset/dataset.h"

namespace mattTorch {

struct GeneratedData {
  Tensor inputs;
  Tensor labels;
};

class SyntheticRegressionDataset : public dataset {
 private:
  SyntheticRegressionDataset(GeneratedData data, int batchSize)
      : dataset(batchSize, data.inputs, data.labels) {};

  static GeneratedData generate(Tensor weight, Tensor bias, double noise,
                                int numTrain) {
    Tensor inputs({numTrain, weight.getNValues()});
    Tensor error({numTrain, 1});

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution d(0.0, 1.0);

    for (int i{0}; i < inputs.getNValues(); i++) {
      inputs.setValueDirect(i, d(gen));
    }

    for (int i{0}; i < error.getNValues(); i++) {
      error.setValueDirect(i, d(gen) * noise);
    }

    Tensor labels = inputs.matrixMultiply(weight) +
                    bias.broadcast(0, inputs.getDimensions()[0]) + error;
    return {inputs, labels};
  }

 public:
  SyntheticRegressionDataset(Tensor weight, Tensor bias, double noise,
                             int numTrain, int batchSize)
      : SyntheticRegressionDataset(generate(weight, bias, noise, numTrain),
                                   batchSize) {}
};
}  // namespace mattTorch
