#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/elementwise.h>

namespace mattTorch::tensor::kernels::cpu {
void elementwiseAdd(const double* __restrict lhs, const double* __restrict rhs,
                    double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(lhs + i);
    __m256d b = _mm256_load_pd(rhs + i);
    __m256d c = _mm256_add_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] + rhs[i];
  }
}

void elementwiseSubtract(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(lhs + i);
    __m256d b = _mm256_load_pd(rhs + i);
    __m256d c = _mm256_sub_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] - rhs[i];
  }
}

void elementwiseMultiplication(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(lhs + i);
    __m256d b = _mm256_load_pd(rhs + i);
    __m256d c = _mm256_mul_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] * rhs[i];
  }
}

void elementwiseDivision(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_load_pd(lhs + i);
    __m256d b = _mm256_load_pd(rhs + i);
    __m256d c = _mm256_div_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] / rhs[i];
  }
}
void reductionSum(const double* __restrict input, double* __restrict result,
                  int outerSize, int reduceSize, int innerSize) {
  for (int outer = 0; outer < outerSize; outer++) {
    for (int inner = 0; inner < innerSize; inner++) {
      double acc = 0.0;
      for (int r = 0; r < reduceSize; r++) {
        int idx = outer * reduceSize * innerSize + r * innerSize + inner;
        acc += input[idx];
      }
      result[outer * innerSize + inner] = acc;
    }
  }
}
void inplaceElementwiseAdd(double* __restrict lhs, const double* __restrict rhs,
                           int nValues) {
  for (int i = 0; i < nValues - 3; i += 4) {
    __m256d a = _mm256_loadu_pd(rhs + i);
    __m256d b = _mm256_loadu_pd(lhs + i);
    __m256d c = _mm256_add_pd(a, b);
    _mm256_storeu_pd(lhs + i, c);
  }

  for (int i = nValues - (nValues % 4); i < nValues; i++) {
    lhs[i] += rhs[i];
  }
}
}  // namespace mattTorch::tensor::kernels::cpu
