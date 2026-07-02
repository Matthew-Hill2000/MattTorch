#include <immintrin.h>
#include <mattTorch/optimiser/sgd/sgd.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <memory>
#include <vector>

namespace mattTorch {
SGD::SGD(std::vector<std::shared_ptr<Tensor>> parameters, double learningRate,
         double weightDecay)
    : parameters{parameters},
      learningRate{learningRate},
      weightDecay{weightDecay} {}

void SGD::updateParameters() {
  __m256d learningRateVec =
      _mm256_set_pd(learningRate, learningRate, learningRate, learningRate);

  __m256d weightDecayVec = _mm256_set_pd(
      (1 - learningRate * weightDecay), (1 - learningRate * weightDecay),
      (1 - learningRate * weightDecay), (1 - learningRate * weightDecay));

  for (auto parameter : parameters) {
    double* paramData = parameter->getData();
    double* paramGradData = parameter->getGradientData();

    int i{0};
    for (; i + 3 < parameter->getNValues(); i += 4) {
      __m256d p = _mm256_load_pd(paramData + i);

      __m256d pW = _mm256_mul_pd(weightDecayVec, p);
      __m256d pG = _mm256_load_pd(paramGradData + i);
      __m256d newP = _mm256_fnmadd_pd(learningRateVec, pG, pW);
      _mm256_store_pd(paramData + i, newP);
    }

    for (; i < parameter->getNValues(); i++) {
      paramData[i] = paramData[i] * (1 - learningRate * weightDecay) -
                     learningRate * paramGradData[i];
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
