#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-scalar/tensorScalarElementwise.h>

namespace mattTorch::tensor::kernels::cpu {
void tensorScalarAdd(const double* __restrict tensor, const double scalar,
                     double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);

    __m256d c0 = _mm256_add_pd(a, b0);
    __m256d c1 = _mm256_add_pd(a, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] + scalar;
  }
}

void tensorScalarSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);

    __m256d c0 = _mm256_sub_pd(b0, a);
    __m256d c1 = _mm256_sub_pd(b1, a);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] - scalar;
  }
}

void scalarTensorSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);

    __m256d c0 = _mm256_sub_pd(a, b0);
    __m256d c1 = _mm256_sub_pd(a, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
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
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);

    __m256d c0 = _mm256_mul_pd(a, b0);
    __m256d c1 = _mm256_mul_pd(a, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] * scalar;
  }
}

void tensorScalarDivision(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);

    __m256d c0 = _mm256_div_pd(b0, a);
    __m256d c1 = _mm256_div_pd(b1, a);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = tensor[i] / scalar;
  }
}
void scalarTensorDivision(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues) {
  __m256d a = _mm256_set_pd(scalar, scalar, scalar, scalar);
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);

    __m256d c0 = _mm256_div_pd(a, b0);
    __m256d c1 = _mm256_div_pd(a, b1);

    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
  }
  for (; i < nValues; ++i) {
    result[i] = scalar / tensor[i];
  }
}

void elementwiseExponent(const double* __restrict tensor, const int scalar,
                         double* __restrict result, const int nValues) {
  int i = 0;
  for (; i + 7 < nValues; i += 8) {
    __m256d b0 = _mm256_load_pd(tensor + i);
    __m256d b1 = _mm256_load_pd(tensor + i + 4);
    __m256d c0 = b0;
    __m256d c1 = b1;
    for (int j{1}; j < scalar; j++) {
      c0 = _mm256_mul_pd(c0, b0);
      c1 = _mm256_mul_pd(c1, b1);
    }
    _mm256_stream_pd(result + i, c0);
    _mm256_stream_pd(result + i + 4, c1);
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
