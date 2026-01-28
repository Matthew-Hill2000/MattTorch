#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor/tensorElementwise.h>
#include <sleef.h>

#include <cmath>
#include <cstring>

namespace mattTorch::tensor::kernels::cpu {

void ReLU(double* __restrict tensor, double* __restrict result,
          double* __restrict backwardMask, int nValues) {
  int i{0};
  const __m256d zero = _mm256_setzero_pd();
  const __m256d ones = _mm256_set1_pd(1.0);
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(tensor + i);
    __m256d relu = _mm256_max_pd(a, zero);

    __m256d gt0 = _mm256_cmp_pd(a, zero, _CMP_GT_OQ);
    __m256d mask = _mm256_and_pd(gt0, ones);

    _mm256_store_pd(backwardMask + i, mask);
    _mm256_store_pd(result + i, relu);
  }

  for (; i < nValues; i++) {
    if (tensor[i] > 0.0) {
      result[i] = tensor[i];
      backwardMask[i] = 1.0;
    } else {
      result[i] = 0.0;
      backwardMask[i] = 0.0;
    }
  }
}

void broadcast(double* __restrict tensor, double* __restrict result,
               int blockSize, int numBlocks, int repeatCount) {
  for (int block = 0; block < numBlocks; block++) {
    for (int rep = 0; rep < repeatCount; rep++) {
      std::memcpy(result + (block * repeatCount + rep) * blockSize,
                  tensor + block * blockSize, blockSize * sizeof(double));
    }
  }
}
void tanh(double* __restrict tensor, double* __restrict result, int nValues) {
  int i{0};
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(tensor + i);
    __m256d b = Sleef_tanhd4_u35avx2(a);

    _mm256_store_pd(result + i, b);
  }

  for (; i < nValues; i++) {
    result[i] = std::tanh(tensor[i]);
  }
}
void exponential(double* __restrict tensor, double* __restrict result, int nValues) {
  int i{0};
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(tensor + i);
    __m256d b = Sleef_expd4_u10avx2(a);

    _mm256_store_pd(result + i, b);
  }

  for (; i < nValues; i++) {
    result[i] = std::exp(tensor[i]);
  }
}

void log(double* __restrict tensor, double* __restrict result, int nValues) {
  int i{0};
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(tensor + i);
    __m256d b = Sleef_logd4_u35avx2(a);
    _mm256_store_pd(result + i, b);
  }

  for (; i < nValues; i++) {
    result[i] = std::log(tensor[i]);
  }
}

void mean(double* __restrict tensor, double* __restrict result, int nValues) {
  __m256d mean = _mm256_setzero_pd();
  int i{0};
  for (; i + 3 < nValues; i += 4) {
    mean = _mm256_add_pd(mean, _mm256_load_pd(tensor + i));
  }

  __m128d lower = _mm256_castpd256_pd128(mean);
  __m128d higher = _mm256_extractf128_pd(mean, 1);

  __m128d sum = _mm_add_pd(lower, higher);
  *result = _mm_cvtsd_f64(_mm_hadd_pd(sum, sum));

  for (; i < nValues; i++) {
    *result += tensor[i];
  }

  *result /= nValues;
}
}  // namespace mattTorch::tensor::kernels::cpu
