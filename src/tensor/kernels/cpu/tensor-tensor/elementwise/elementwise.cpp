#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/elementwise.h>

namespace mattTorch::tensor::kernels::cpu {

void elementwiseAdd(const double* __restrict lhs, const double* __restrict rhs,
                    double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d a0 = _mm256_load_pd(lhs + i);
    __m256d b0 = _mm256_load_pd(rhs + i);

    __m256d a1 = _mm256_load_pd(lhs + i + 4);
    __m256d b1 = _mm256_load_pd(rhs + i + 4);

    __m256d c0 = _mm256_add_pd(a0, b0);
    __m256d c1 = _mm256_add_pd(a1, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] + rhs[i];
  }
}

void elementwiseSubtract(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d a0 = _mm256_load_pd(lhs + i);
    __m256d b0 = _mm256_load_pd(rhs + i);

    __m256d a1 = _mm256_load_pd(lhs + i + 4);
    __m256d b1 = _mm256_load_pd(rhs + i + 4);

    __m256d c0 = _mm256_sub_pd(a0, b0);
    __m256d c1 = _mm256_sub_pd(a1, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] - rhs[i];
  }
}

void elementwiseMultiplication(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d a0 = _mm256_load_pd(lhs + i);
    __m256d b0 = _mm256_load_pd(rhs + i);

    __m256d a1 = _mm256_load_pd(lhs + i + 4);
    __m256d b1 = _mm256_load_pd(rhs + i + 4);

    __m256d c0 = _mm256_mul_pd(a0, b0);
    __m256d c1 = _mm256_mul_pd(a1, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = lhs[i] * rhs[i];
  }
}

void elementwiseDivision(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d a0 = _mm256_load_pd(lhs + i);
    __m256d b0 = _mm256_load_pd(rhs + i);

    __m256d a1 = _mm256_load_pd(lhs + i + 4);
    __m256d b1 = _mm256_load_pd(rhs + i + 4);

    __m256d c0 = _mm256_div_pd(a0, b0);
    __m256d c1 = _mm256_div_pd(a1, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
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
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d a0 = _mm256_loadu_pd(rhs + i);
    __m256d b0 = _mm256_loadu_pd(lhs + i);

    __m256d a1 = _mm256_loadu_pd(rhs + i + 4);
    __m256d b1 = _mm256_loadu_pd(lhs + i + 4);

    __m256d c0 = _mm256_add_pd(a0, b0);
    __m256d c1 = _mm256_add_pd(a1, b1);

    _mm256_storeu_pd(lhs + i, c0);
    _mm256_storeu_pd(lhs + i + 4, c1);
  }

  for (; i < nValues; i++) {
    lhs[i] += rhs[i];
  }
}
}  // namespace mattTorch::tensor::kernels::cpu
