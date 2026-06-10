#include <mattTorch/optimiser/sgd/sgd.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <memory>
#include <vector>

namespace mattTorch {
SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, double learningRate)
    : parameters{parameters}, learningRate{learningRate} {}

void SGD::updateParameters() {
  __m256d learningRateVec =
      _mm256_set_pd(learningRate, learningRate, learningRate, learningRate);

  for (auto parameter : parameters) {
    double* paramData = parameter->getData();
    double* paramGradData = parameter->getGradientData();

    int i{0};
    for (; i + 3 < parameter->getNValues(); i += 4) {
      __m256d p = _mm256_load_pd(paramData + i);
      __m256d pG = _mm256_load_pd(paramGradData + i);
      __m256d newP = _mm256_fnmadd_pd(learningRateVec, pG, p);
      _mm256_store_pd(paramData + i, newP);
    }

    for (; i < parameter->getNValues(); i++) {
      paramData[i] -= learningRate * paramGradData[i];
    }
  }
}

void SGD::zeroGrad() {
  for (auto parameter : parameters) {
    double* paramGradData = parameter->getGradientData();
    std::fill(paramGradData, paramGradData + parameter->getNValues(), 0);
  }
}

}  // namespace mattTorch
