#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <algorithm>

constexpr int BIGBLOCKSIZE_I = 128;
constexpr int BIGBLOCKSIZE_J = 128;
constexpr int BIGBLOCKSIZE_K = 128;

namespace mattTorch::tensor::kernels::cpu {

void inline rhsBigPack(double* block, const double* rhs, const int lhsCol,
                       const int rhsCol, const int rhsCols, const int kMax,
                       const int jMax) {
  int j{0};
  for (; j + 7 < jMax; j += 8) {
    for (int k{0}; k < kMax; k++) {
      __m256d src0 = _mm256_loadu_pd(&rhs[(lhsCol + k) * rhsCols + rhsCol + j]);
      __m256d src1 =
          _mm256_loadu_pd(&rhs[(lhsCol + k) * rhsCols + rhsCol + j + 4]);
      _mm256_store_pd(&block[j * kMax + 8 * k + 0], src0);
      _mm256_store_pd(&block[j * kMax + 8 * k + 4], src1);
    }
  }

  if (j + 3 < jMax) {
    for (int k{0}; k < kMax; k++) {
      __m256d src = _mm256_loadu_pd(&rhs[(lhsCol + k) * rhsCols + rhsCol + j]);
      _mm256_store_pd(&block[j * kMax + 4 * k], src);
    }
    j += 4;
  }

  for (; j < jMax; j++) {
    for (int k{0}; k < kMax; k++) {
      block[j * kMax + k] = rhs[(lhsCol + k) * rhsCols + rhsCol + j];
    }
  }
}

void inline lhsBigPack(double* block, const double* lhs, const int lhsRow,
                       const int lhsCol, const int lhsCols, const int iMax,
                       const int kMax) {
  int i{0};
  for (; i + 5 < iMax; i += 6) {
    for (int k{0}; k < kMax; k++) {
      block[i * kMax + 6 * k] = lhs[(lhsRow + i) * lhsCols + lhsCol + k];
      block[i * kMax + 6 * k + 1] =
          lhs[(lhsRow + i + 1) * lhsCols + lhsCol + k];
      block[i * kMax + 6 * k + 2] =
          lhs[(lhsRow + i + 2) * lhsCols + lhsCol + k];
      block[i * kMax + 6 * k + 3] =
          lhs[(lhsRow + i + 3) * lhsCols + lhsCol + k];
      block[i * kMax + 6 * k + 4] =
          lhs[(lhsRow + i + 4) * lhsCols + lhsCol + k];
      block[i * kMax + 6 * k + 5] =
          lhs[(lhsRow + i + 5) * lhsCols + lhsCol + k];
    }
  }

  for (; i < iMax; i++) {
    for (int k{0}; k < kMax; k++) {
      block[i * kMax + k] = lhs[(lhsRow + i) * lhsCols + lhsCol + k];
    }
  }
}

void inline matrixMultMicroKernel(const double* lhsBlock,
                                  const double* rhsBlock, double* result,
                                  int lhsRow, int rhsCol, const int rhsCols,
                                  int iMax, int jMax, int kMax) {
  for (int i{0}; i + 5 < iMax; i += 6) {
    int j = 0;
    for (; j + 7 < jMax; j += 8) {
      __m256d c00 =
          _mm256_loadu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j]);
      __m256d c10 =
          _mm256_loadu_pd(&result[(lhsRow + i + 1) * rhsCols + rhsCol + j]);
      __m256d c20 =
          _mm256_loadu_pd(&result[(lhsRow + i + 2) * rhsCols + rhsCol + j]);
      __m256d c30 =
          _mm256_loadu_pd(&result[(lhsRow + i + 3) * rhsCols + rhsCol + j]);
      __m256d c40 =
          _mm256_loadu_pd(&result[(lhsRow + i + 4) * rhsCols + rhsCol + j]);
      __m256d c50 =
          _mm256_loadu_pd(&result[(lhsRow + i + 5) * rhsCols + rhsCol + j]);
      __m256d c04 =
          _mm256_loadu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j + 4]);
      __m256d c14 =
          _mm256_loadu_pd(&result[(lhsRow + i + 1) * rhsCols + rhsCol + j + 4]);
      __m256d c24 =
          _mm256_loadu_pd(&result[(lhsRow + i + 2) * rhsCols + rhsCol + j + 4]);
      __m256d c34 =
          _mm256_loadu_pd(&result[(lhsRow + i + 3) * rhsCols + rhsCol + j + 4]);
      __m256d c44 =
          _mm256_loadu_pd(&result[(lhsRow + i + 4) * rhsCols + rhsCol + j + 4]);
      __m256d c54 =
          _mm256_loadu_pd(&result[(lhsRow + i + 5) * rhsCols + rhsCol + j + 4]);

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
      _mm256_storeu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j], c00);
      _mm256_storeu_pd(&result[(lhsRow + i + 1) * rhsCols + rhsCol + j], c10);
      _mm256_storeu_pd(&result[(lhsRow + i + 2) * rhsCols + rhsCol + j], c20);
      _mm256_storeu_pd(&result[(lhsRow + i + 3) * rhsCols + rhsCol + j], c30);
      _mm256_storeu_pd(&result[(lhsRow + i + 4) * rhsCols + rhsCol + j], c40);
      _mm256_storeu_pd(&result[(lhsRow + i + 5) * rhsCols + rhsCol + j], c50);
      _mm256_storeu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j + 4], c04);
      _mm256_storeu_pd(&result[(lhsRow + i + 1) * rhsCols + rhsCol + j + 4],
                       c14);
      _mm256_storeu_pd(&result[(lhsRow + i + 2) * rhsCols + rhsCol + j + 4],
                       c24);
      _mm256_storeu_pd(&result[(lhsRow + i + 3) * rhsCols + rhsCol + j + 4],
                       c34);

      _mm256_storeu_pd(&result[(lhsRow + i + 4) * rhsCols + rhsCol + j + 4],
                       c44);
      _mm256_storeu_pd(&result[(lhsRow + i + 5) * rhsCols + rhsCol + j + 4],
                       c54);
    }

    if (j + 3 < jMax) {
      __m256d acc0 =
          _mm256_loadu_pd(&result[(lhsRow + i + 0) * rhsCols + rhsCol + j]);
      __m256d acc1 =
          _mm256_loadu_pd(&result[(lhsRow + i + 1) * rhsCols + rhsCol + j]);
      __m256d acc2 =
          _mm256_loadu_pd(&result[(lhsRow + i + 2) * rhsCols + rhsCol + j]);
      __m256d acc3 =
          _mm256_loadu_pd(&result[(lhsRow + i + 3) * rhsCols + rhsCol + j]);
      __m256d acc4 =
          _mm256_loadu_pd(&result[(lhsRow + i + 4) * rhsCols + rhsCol + j]);
      __m256d acc5 =
          _mm256_loadu_pd(&result[(lhsRow + i + 5) * rhsCols + rhsCol + j]);

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

      _mm256_storeu_pd(&result[(lhsRow + i + 0) * rhsCols + rhsCol + j], acc0);
      _mm256_storeu_pd(&result[(lhsRow + i + 1) * rhsCols + rhsCol + j], acc1);
      _mm256_storeu_pd(&result[(lhsRow + i + 2) * rhsCols + rhsCol + j], acc2);
      _mm256_storeu_pd(&result[(lhsRow + i + 3) * rhsCols + rhsCol + j], acc3);
      _mm256_storeu_pd(&result[(lhsRow + i + 4) * rhsCols + rhsCol + j], acc4);
      _mm256_storeu_pd(&result[(lhsRow + i + 5) * rhsCols + rhsCol + j], acc5);
      j += 4;
    }

    for (; j < jMax; ++j) {
      double c0 = result[(lhsRow + i + 0) * rhsCols + rhsCol + j];
      double c1 = result[(lhsRow + i + 1) * rhsCols + rhsCol + j];
      double c2 = result[(lhsRow + i + 2) * rhsCols + rhsCol + j];
      double c3 = result[(lhsRow + i + 3) * rhsCols + rhsCol + j];
      double c4 = result[(lhsRow + i + 4) * rhsCols + rhsCol + j];
      double c5 = result[(lhsRow + i + 5) * rhsCols + rhsCol + j];
      for (int k = 0; k < kMax; ++k) {
        double b = rhsBlock[j * kMax + k];
        c0 += lhsBlock[i * kMax + 6 * k + 0] * b;
        c1 += lhsBlock[i * kMax + 6 * k + 1] * b;
        c2 += lhsBlock[i * kMax + 6 * k + 2] * b;
        c3 += lhsBlock[i * kMax + 6 * k + 3] * b;
        c4 += lhsBlock[i * kMax + 6 * k + 4] * b;
        c5 += lhsBlock[i * kMax + 6 * k + 5] * b;
      }
      result[(lhsRow + i + 0) * rhsCols + rhsCol + j] = c0;
      result[(lhsRow + i + 1) * rhsCols + rhsCol + j] = c1;
      result[(lhsRow + i + 2) * rhsCols + rhsCol + j] = c2;
      result[(lhsRow + i + 3) * rhsCols + rhsCol + j] = c3;
      result[(lhsRow + i + 4) * rhsCols + rhsCol + j] = c4;
      result[(lhsRow + i + 5) * rhsCols + rhsCol + j] = c5;
    }
  }

  for (int i = (iMax / 6) * 6; i < iMax; ++i) {
    int j = 0;
    for (; j + 7 < jMax; j += 8) {
      __m256d acc0 =
          _mm256_loadu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j]);
      __m256d acc1 =
          _mm256_loadu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j + 4]);
      for (int k = 0; k < kMax; ++k) {
        __m256d b0 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 0]);
        __m256d b1 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 4]);
        __m256d a = _mm256_broadcast_sd(&lhsBlock[i * kMax + k]);
        acc0 = _mm256_fmadd_pd(a, b0, acc0);
        acc1 = _mm256_fmadd_pd(a, b1, acc1);
      }
      _mm256_storeu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j], acc0);
      _mm256_storeu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j + 4], acc1);
    }
    if (j + 3 < jMax) {
      __m256d acc =
          _mm256_loadu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j]);
      for (int k = 0; k < kMax; ++k) {
        __m256d b = _mm256_loadu_pd(&rhsBlock[j * kMax + 4 * k]);
        __m256d a = _mm256_broadcast_sd(&lhsBlock[i * kMax + k]);
        acc = _mm256_fmadd_pd(a, b, acc);
      }
      _mm256_storeu_pd(&result[(lhsRow + i) * rhsCols + rhsCol + j], acc);
      j += 4;
    }
    for (; j < jMax; ++j) {
      double c = result[(lhsRow + i) * rhsCols + rhsCol + j];
      for (int k = 0; k < kMax; ++k) {
        c += lhsBlock[i * kMax + k] * rhsBlock[j * kMax + k];
      }
      result[(lhsRow + i) * rhsCols + rhsCol + j] = c;
    }
  }
}

void matrixMultBlockPackedVector(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsCols) {
#pragma omp parallel
  {
    double* rhsBigBlock = nullptr;
    posix_memalign(reinterpret_cast<void**>(&rhsBigBlock), 64,
                   BIGBLOCKSIZE_K * BIGBLOCKSIZE_J * sizeof(double));

    double* lhsBigBlock = nullptr;
    posix_memalign(reinterpret_cast<void**>(&lhsBigBlock), 64,
                   BIGBLOCKSIZE_I * BIGBLOCKSIZE_K * sizeof(double));

#pragma omp for collapse(2) schedule(dynamic)
    for (int rhsCol = 0; rhsCol < rhsCols; rhsCol += BIGBLOCKSIZE_J) {
      for (int lhsRow = 0; lhsRow < lhsRows; lhsRow += BIGBLOCKSIZE_I) {
        for (int lhsCol = 0; lhsCol < lhsCols; lhsCol += BIGBLOCKSIZE_K) {
          int iMax = std::min(BIGBLOCKSIZE_I, lhsRows - lhsRow);
          int jMax = std::min(BIGBLOCKSIZE_J, rhsCols - rhsCol);
          int kMax = std::min(BIGBLOCKSIZE_K, lhsCols - lhsCol);

          rhsBigPack(rhsBigBlock, rhs, lhsCol, rhsCol, rhsCols, kMax, jMax);
          lhsBigPack(lhsBigBlock, lhs, lhsRow, lhsCol, lhsCols, iMax, kMax);

          matrixMultMicroKernel(lhsBigBlock, rhsBigBlock, result, lhsRow,
                                rhsCol, rhsCols, iMax, jMax, kMax);
        }
      }
    }
    free(rhsBigBlock);
    free(lhsBigBlock);
  }
}
}  // namespace mattTorch::tensor::kernels::cpu
