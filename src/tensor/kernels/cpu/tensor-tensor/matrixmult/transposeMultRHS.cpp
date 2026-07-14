// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// bugprone-easily-swappable-parameters)
#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <algorithm>

namespace mattTorch::tensor::kernels::cpu {

namespace {

constexpr int BIGBLOCKSIZE_M = 128;
constexpr int BIGBLOCKSIZE_N = 128;
constexpr int BIGBLOCKSIZE_K = 128;

void packRhs(double* block, const double* rhs, int kBase, int nBase,
             int rowStride, int kMax, int nMax) {
  int j{0};
  for (; j + 7 < nMax; j += 8) {
    for (int k{0}; k < kMax; k++) {
      block[j * kMax + 8 * k + 0] =
          rhs[(nBase + j + 0) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 1] =
          rhs[(nBase + j + 1) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 2] =
          rhs[(nBase + j + 2) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 3] =
          rhs[(nBase + j + 3) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 4] =
          rhs[(nBase + j + 4) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 5] =
          rhs[(nBase + j + 5) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 6] =
          rhs[(nBase + j + 6) * rowStride + kBase + k];
      block[j * kMax + 8 * k + 7] =
          rhs[(nBase + j + 7) * rowStride + kBase + k];
    }
  }

  if (j + 3 < nMax) {
    for (int k{0}; k < kMax; k++) {
      block[j * kMax + 4 * k + 0] =
          rhs[(nBase + j + 0) * rowStride + kBase + k];
      block[j * kMax + 4 * k + 1] =
          rhs[(nBase + j + 1) * rowStride + kBase + k];
      block[j * kMax + 4 * k + 2] =
          rhs[(nBase + j + 2) * rowStride + kBase + k];
      block[j * kMax + 4 * k + 3] =
          rhs[(nBase + j + 3) * rowStride + kBase + k];
    }
    j += 4;
  }

  for (; j < nMax; j++) {
    for (int k{0}; k < kMax; k++) {
      block[j * kMax + k] = rhs[(nBase + j) * rowStride + kBase + k];
    }
  }
}

void packLhs(double* block, const double* lhs, int mBase, int kBase,
             int rowStride, int mMax, int kMax) {
  int i{0};
  for (; i + 5 < mMax; i += 6) {
    for (int k{0}; k < kMax; k++) {
      block[i * kMax + 6 * k + 0] =
          lhs[(mBase + i + 0) * rowStride + kBase + k];
      block[i * kMax + 6 * k + 1] =
          lhs[(mBase + i + 1) * rowStride + kBase + k];
      block[i * kMax + 6 * k + 2] =
          lhs[(mBase + i + 2) * rowStride + kBase + k];
      block[i * kMax + 6 * k + 3] =
          lhs[(mBase + i + 3) * rowStride + kBase + k];
      block[i * kMax + 6 * k + 4] =
          lhs[(mBase + i + 4) * rowStride + kBase + k];
      block[i * kMax + 6 * k + 5] =
          lhs[(mBase + i + 5) * rowStride + kBase + k];
    }
  }

  for (; i < mMax; i++) {
    for (int k{0}; k < kMax; k++) {
      block[i * kMax + k] = lhs[(mBase + i) * rowStride + kBase + k];
    }
  }
}

void microKernel(const double* lhsBlock, const double* rhsBlock, double* result,
                 int rowBase, int colBase, int outStride, int mMax, int nMax,
                 int kMax) {
  for (int i{0}; i + 5 < mMax; i += 6) {
    int j = 0;
    for (; j + 7 < nMax; j += 8) {
      __m256d c00 =
          _mm256_loadu_pd(&result[(rowBase + i) * outStride + colBase + j]);
      __m256d c10 =
          _mm256_loadu_pd(&result[(rowBase + i + 1) * outStride + colBase + j]);
      __m256d c20 =
          _mm256_loadu_pd(&result[(rowBase + i + 2) * outStride + colBase + j]);
      __m256d c30 =
          _mm256_loadu_pd(&result[(rowBase + i + 3) * outStride + colBase + j]);
      __m256d c40 =
          _mm256_loadu_pd(&result[(rowBase + i + 4) * outStride + colBase + j]);
      __m256d c50 =
          _mm256_loadu_pd(&result[(rowBase + i + 5) * outStride + colBase + j]);
      __m256d c04 =
          _mm256_loadu_pd(&result[(rowBase + i) * outStride + colBase + j + 4]);
      __m256d c14 = _mm256_loadu_pd(
          &result[(rowBase + i + 1) * outStride + colBase + j + 4]);
      __m256d c24 = _mm256_loadu_pd(
          &result[(rowBase + i + 2) * outStride + colBase + j + 4]);
      __m256d c34 = _mm256_loadu_pd(
          &result[(rowBase + i + 3) * outStride + colBase + j + 4]);
      __m256d c44 = _mm256_loadu_pd(
          &result[(rowBase + i + 4) * outStride + colBase + j + 4]);
      __m256d c54 = _mm256_loadu_pd(
          &result[(rowBase + i + 5) * outStride + colBase + j + 4]);

      for (int k{0}; k < kMax; k++) {
        __m256d b0 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 0]);
        __m256d b1 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 4]);

        __m256d a0 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 0]);
        __m256d a1 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 1]);
        __m256d a2 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 2]);
        __m256d a3 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 3]);
        __m256d a4 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 4]);
        __m256d a5 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 5]);

        c00 = _mm256_fmadd_pd(a0, b0, c00);
        c10 = _mm256_fmadd_pd(a1, b0, c10);
        c20 = _mm256_fmadd_pd(a2, b0, c20);
        c30 = _mm256_fmadd_pd(a3, b0, c30);
        c40 = _mm256_fmadd_pd(a4, b0, c40);
        c50 = _mm256_fmadd_pd(a5, b0, c50);

        c04 = _mm256_fmadd_pd(a0, b1, c04);
        c14 = _mm256_fmadd_pd(a1, b1, c14);
        c24 = _mm256_fmadd_pd(a2, b1, c24);
        c34 = _mm256_fmadd_pd(a3, b1, c34);
        c44 = _mm256_fmadd_pd(a4, b1, c44);
        c54 = _mm256_fmadd_pd(a5, b1, c54);
      }
      _mm256_storeu_pd(&result[(rowBase + i) * outStride + colBase + j], c00);
      _mm256_storeu_pd(&result[(rowBase + i + 1) * outStride + colBase + j],
                       c10);
      _mm256_storeu_pd(&result[(rowBase + i + 2) * outStride + colBase + j],
                       c20);
      _mm256_storeu_pd(&result[(rowBase + i + 3) * outStride + colBase + j],
                       c30);
      _mm256_storeu_pd(&result[(rowBase + i + 4) * outStride + colBase + j],
                       c40);
      _mm256_storeu_pd(&result[(rowBase + i + 5) * outStride + colBase + j],
                       c50);
      _mm256_storeu_pd(&result[(rowBase + i) * outStride + colBase + j + 4],
                       c04);
      _mm256_storeu_pd(&result[(rowBase + i + 1) * outStride + colBase + j + 4],
                       c14);
      _mm256_storeu_pd(&result[(rowBase + i + 2) * outStride + colBase + j + 4],
                       c24);
      _mm256_storeu_pd(&result[(rowBase + i + 3) * outStride + colBase + j + 4],
                       c34);
      _mm256_storeu_pd(&result[(rowBase + i + 4) * outStride + colBase + j + 4],
                       c44);
      _mm256_storeu_pd(&result[(rowBase + i + 5) * outStride + colBase + j + 4],
                       c54);
    }

    if (j + 3 < nMax) {
      __m256d acc0 =
          _mm256_loadu_pd(&result[(rowBase + i + 0) * outStride + colBase + j]);
      __m256d acc1 =
          _mm256_loadu_pd(&result[(rowBase + i + 1) * outStride + colBase + j]);
      __m256d acc2 =
          _mm256_loadu_pd(&result[(rowBase + i + 2) * outStride + colBase + j]);
      __m256d acc3 =
          _mm256_loadu_pd(&result[(rowBase + i + 3) * outStride + colBase + j]);
      __m256d acc4 =
          _mm256_loadu_pd(&result[(rowBase + i + 4) * outStride + colBase + j]);
      __m256d acc5 =
          _mm256_loadu_pd(&result[(rowBase + i + 5) * outStride + colBase + j]);

      for (int k = 0; k < kMax; ++k) {
        __m256d b = _mm256_loadu_pd(&rhsBlock[kMax * j + 4 * k]);
        __m256d a0 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 0]);
        __m256d a1 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 1]);
        __m256d a2 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 2]);
        __m256d a3 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 3]);
        __m256d a4 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 4]);
        __m256d a5 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 5]);
        acc0 = _mm256_fmadd_pd(a0, b, acc0);
        acc1 = _mm256_fmadd_pd(a1, b, acc1);
        acc2 = _mm256_fmadd_pd(a2, b, acc2);
        acc3 = _mm256_fmadd_pd(a3, b, acc3);
        acc4 = _mm256_fmadd_pd(a4, b, acc4);
        acc5 = _mm256_fmadd_pd(a5, b, acc5);
      }

      _mm256_storeu_pd(&result[(rowBase + i + 0) * outStride + colBase + j],
                       acc0);
      _mm256_storeu_pd(&result[(rowBase + i + 1) * outStride + colBase + j],
                       acc1);
      _mm256_storeu_pd(&result[(rowBase + i + 2) * outStride + colBase + j],
                       acc2);
      _mm256_storeu_pd(&result[(rowBase + i + 3) * outStride + colBase + j],
                       acc3);
      _mm256_storeu_pd(&result[(rowBase + i + 4) * outStride + colBase + j],
                       acc4);
      _mm256_storeu_pd(&result[(rowBase + i + 5) * outStride + colBase + j],
                       acc5);
      j += 4;
    }

    for (; j < nMax; ++j) {
      double c0 = result[(rowBase + i + 0) * outStride + colBase + j];
      double c1 = result[(rowBase + i + 1) * outStride + colBase + j];
      double c2 = result[(rowBase + i + 2) * outStride + colBase + j];
      double c3 = result[(rowBase + i + 3) * outStride + colBase + j];
      double c4 = result[(rowBase + i + 4) * outStride + colBase + j];
      double c5 = result[(rowBase + i + 5) * outStride + colBase + j];
      for (int k = 0; k < kMax; ++k) {
        double b = rhsBlock[j * kMax + k];
        c0 += lhsBlock[i * kMax + 6 * k + 0] * b;
        c1 += lhsBlock[i * kMax + 6 * k + 1] * b;
        c2 += lhsBlock[i * kMax + 6 * k + 2] * b;
        c3 += lhsBlock[i * kMax + 6 * k + 3] * b;
        c4 += lhsBlock[i * kMax + 6 * k + 4] * b;
        c5 += lhsBlock[i * kMax + 6 * k + 5] * b;
      }
      result[(rowBase + i + 0) * outStride + colBase + j] = c0;
      result[(rowBase + i + 1) * outStride + colBase + j] = c1;
      result[(rowBase + i + 2) * outStride + colBase + j] = c2;
      result[(rowBase + i + 3) * outStride + colBase + j] = c3;
      result[(rowBase + i + 4) * outStride + colBase + j] = c4;
      result[(rowBase + i + 5) * outStride + colBase + j] = c5;
    }
  }

  for (int i = (mMax / 6) * 6; i < mMax; ++i) {
    int j = 0;
    for (; j + 7 < nMax; j += 8) {
      __m256d acc0 =
          _mm256_loadu_pd(&result[(rowBase + i) * outStride + colBase + j]);
      __m256d acc1 =
          _mm256_loadu_pd(&result[(rowBase + i) * outStride + colBase + j + 4]);
      for (int k = 0; k < kMax; ++k) {
        __m256d b0 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 0]);
        __m256d b1 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 4]);
        __m256d a = _mm256_broadcast_sd(&lhsBlock[i * kMax + k]);
        acc0 = _mm256_fmadd_pd(a, b0, acc0);
        acc1 = _mm256_fmadd_pd(a, b1, acc1);
      }
      _mm256_storeu_pd(&result[(rowBase + i) * outStride + colBase + j], acc0);
      _mm256_storeu_pd(&result[(rowBase + i) * outStride + colBase + j + 4],
                       acc1);
    }
    if (j + 3 < nMax) {
      __m256d acc =
          _mm256_loadu_pd(&result[(rowBase + i) * outStride + colBase + j]);
      for (int k = 0; k < kMax; ++k) {
        __m256d b = _mm256_loadu_pd(&rhsBlock[j * kMax + 4 * k]);
        __m256d a = _mm256_broadcast_sd(&lhsBlock[i * kMax + k]);
        acc = _mm256_fmadd_pd(a, b, acc);
      }
      _mm256_storeu_pd(&result[(rowBase + i) * outStride + colBase + j], acc);
      j += 4;
    }
    for (; j < nMax; ++j) {
      double c = result[(rowBase + i) * outStride + colBase + j];
      for (int k = 0; k < kMax; ++k) {
        c += lhsBlock[i * kMax + k] * rhsBlock[j * kMax + k];
      }
      result[(rowBase + i) * outStride + colBase + j] = c;
    }
  }
}

}  // namespace

void transposeMultBlockVectorRHS(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsRows) {
  const int M = lhsRows;
  const int N = rhsRows;
  const int K = lhsCols;

#pragma omp parallel
  {
    double* rhsBigBlock = nullptr;
    posix_memalign(reinterpret_cast<void**>(&rhsBigBlock), 64,
                   BIGBLOCKSIZE_K * BIGBLOCKSIZE_N * sizeof(double));

    double* lhsBigBlock = nullptr;
    posix_memalign(reinterpret_cast<void**>(&lhsBigBlock), 64,
                   BIGBLOCKSIZE_M * BIGBLOCKSIZE_K * sizeof(double));

#pragma omp for collapse(2) schedule(dynamic)
    for (int nBlock = 0; nBlock < N; nBlock += BIGBLOCKSIZE_N) {
      for (int mBlock = 0; mBlock < M; mBlock += BIGBLOCKSIZE_M) {
        for (int kBlock = 0; kBlock < K; kBlock += BIGBLOCKSIZE_K) {
          int mMax = std::min(BIGBLOCKSIZE_M, M - mBlock);
          int nMax = std::min(BIGBLOCKSIZE_N, N - nBlock);
          int kMax = std::min(BIGBLOCKSIZE_K, K - kBlock);

          packRhs(rhsBigBlock, rhs, kBlock, nBlock, lhsCols, kMax, nMax);
          packLhs(lhsBigBlock, lhs, mBlock, kBlock, lhsCols, mMax, kMax);

          microKernel(lhsBigBlock, rhsBigBlock, result, mBlock, nBlock, N, mMax,
                      nMax, kMax);
        }
      }
    }
    free(rhsBigBlock);
    free(lhsBigBlock);
  }
}

}  // namespace mattTorch::tensor::kernels::cpu
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// bugprone-easily-swappable-parameters)
