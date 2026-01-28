#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-scalar/tensorScalarElementwise.h>

namespace mattTorch::tensor::kernels::cpu {
void tensorScalarAdd(const double* __restrict tensor, const double scalar,
                     double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_add_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] + scalar;
  }
}

void tensorScalarSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_sub_pd(b, a);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] - scalar;
  }
}

void scalarTensorSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_sub_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = scalar - tensor[i];
  }
}
void tensorScalarMultiplication(const double* __restrict tensor,
                                const double scalar, double* __restrict result,
                                const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_mul_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] * scalar;
  }
}

void tensorScalarDivision(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_div_pd(b, a);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] / scalar;
  }
}
void scalarTensorDivision(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_div_pd(a, b);
    _mm256_store_pd(result + i, c);
  }
  for (; i < nValues; ++i) {
    result[i] = scalar / tensor[i];
  }
}

void elementwiseExponent(const double* __restrict tensor, const int scalar,
                         double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 3 < nValues; i += 4) {
    __m256d b = _mm256_load_pd(tensor + i);
    __m256d c = _mm256_load_pd(tensor + i);
    for (int j{1}; j < scalar; j++) {
      c = _mm256_mul_pd(c, b);
    }
    _mm256_store_pd(result + i, c);
  }

  for (; i < nValues; i++) {
    double resultD{1};
    for (int j{0}; j < scalar; j++) {
      resultD *= tensor[i];
    }
    result[i] = resultD;
  }
}
}  // namespace mattTorch::tensor::kernels::cpu
