// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// bugprone-easily-swappable-parameters)
#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>

namespace mattTorch::tensor::kernels::cpu {

constexpr int BIGBLOCKSIZE_I = 128;
constexpr int BIGBLOCKSIZE_J = 128;
constexpr int BIGBLOCKSIZE_K = 128;

void inline rhsTranspose(double* block, const double* rhs, const int lhsCol,
                         const int rhsCol, const int rhsCols, const int kMax,
                         const int jMax) {
  for (int x = 0; x < kMax; x++) {
    for (int y = 0; y < jMax; y++) {
      block[y * kMax + x] = rhs[(lhsCol + x) * rhsCols + rhsCol + y];
    }
  }
}

void inline matrixMultMicroKernel(const double* lhs,
                                  const double* rhsBlockTranspose,
                                  double* result, int lhsRow, int rhsCol,
                                  int lhsCol, const int rhsCols,
                                  const int lhsCols, int iMax, int jMax,
                                  int kMax) {
  for (int x = 0; x < iMax; x += 2) {
    for (int y = 0; y < jMax; y += 4) {
      __m256d acc00 = _mm256_setzero_pd();
      __m256d acc01 = _mm256_setzero_pd();
      __m256d acc02 = _mm256_setzero_pd();
      __m256d acc03 = _mm256_setzero_pd();
      __m256d acc10 = _mm256_setzero_pd();
      __m256d acc11 = _mm256_setzero_pd();
      __m256d acc12 = _mm256_setzero_pd();
      __m256d acc13 = _mm256_setzero_pd();

      for (int z = 0; z < kMax; z += 4) {
        __m256d lhs0 =
            _mm256_loadu_pd(&lhs[(lhsRow + x) * lhsCols + lhsCol + z]);
        __m256d lhs1 =
            _mm256_loadu_pd(&lhs[(lhsRow + x + 1) * lhsCols + lhsCol + z]);

        __m256d rhs0 = _mm256_loadu_pd(&rhsBlockTranspose[y * kMax + z]);
        __m256d rhs1 = _mm256_loadu_pd(&rhsBlockTranspose[(y + 1) * kMax + z]);
        __m256d rhs2 = _mm256_loadu_pd(&rhsBlockTranspose[(y + 2) * kMax + z]);
        __m256d rhs3 = _mm256_loadu_pd(&rhsBlockTranspose[(y + 3) * kMax + z]);

        acc00 = _mm256_fmadd_pd(lhs0, rhs0, acc00);
        acc01 = _mm256_fmadd_pd(lhs0, rhs1, acc01);
        acc02 = _mm256_fmadd_pd(lhs0, rhs2, acc02);
        acc03 = _mm256_fmadd_pd(lhs0, rhs3, acc03);
        acc10 = _mm256_fmadd_pd(lhs1, rhs0, acc10);
        acc11 = _mm256_fmadd_pd(lhs1, rhs1, acc11);
        acc12 = _mm256_fmadd_pd(lhs1, rhs2, acc12);
        acc13 = _mm256_fmadd_pd(lhs1, rhs3, acc13);
      }

      __m256d hadd01_0 = _mm256_hadd_pd(acc00, acc01);
      __m256d hadd23_0 = _mm256_hadd_pd(acc02, acc03);

      __m128d lo01_0 = _mm256_castpd256_pd128(hadd01_0);
      __m128d hi01_0 = _mm256_extractf128_pd(hadd01_0, 1);
      __m128d sum01_0 = _mm_add_pd(lo01_0, hi01_0);

      __m128d lo23_0 = _mm256_castpd256_pd128(hadd23_0);
      __m128d hi23_0 = _mm256_extractf128_pd(hadd23_0, 1);
      __m128d sum23_0 = _mm_add_pd(lo23_0, hi23_0);

      __m256d row0_results = _mm256_set_m128d(sum23_0, sum01_0);

      __m256d existing0 =
          _mm256_loadu_pd(&result[(lhsRow + x) * rhsCols + rhsCol + y]);
      _mm256_storeu_pd(&result[(lhsRow + x) * rhsCols + rhsCol + y],
                       _mm256_add_pd(existing0, row0_results));

      __m256d hadd01_1 = _mm256_hadd_pd(acc10, acc11);
      __m256d hadd23_1 = _mm256_hadd_pd(acc12, acc13);

      __m128d lo01_1 = _mm256_castpd256_pd128(hadd01_1);
      __m128d hi01_1 = _mm256_extractf128_pd(hadd01_1, 1);
      __m128d sum01_1 = _mm_add_pd(lo01_1, hi01_1);

      __m128d lo23_1 = _mm256_castpd256_pd128(hadd23_1);
      __m128d hi23_1 = _mm256_extractf128_pd(hadd23_1, 1);
      __m128d sum23_1 = _mm_add_pd(lo23_1, hi23_1);

      __m256d row1_results = _mm256_set_m128d(sum23_1, sum01_1);
      __m256d existing1 =
          _mm256_loadu_pd(&result[(lhsRow + x + 1) * rhsCols + rhsCol + y]);
      _mm256_storeu_pd(&result[(lhsRow + x + 1) * rhsCols + rhsCol + y],
                       _mm256_add_pd(existing1, row1_results));
    }
  }
}

void inline matrixMultMicroKernelITail(const double* lhs,
                                       const double* rhsBlockTranspose,
                                       double* result, int lhsRow, int rhsCol,
                                       int lhsCol, const int rhsCols,
                                       const int lhsCols, int iMax, int jMax,
                                       int kMax) {
  for (int i = lhsRow; i < lhsRow + iMax; i++) {
    for (int y = 0; y < jMax; y += 4) {
      __m256d acc00 = _mm256_setzero_pd();
      __m256d acc01 = _mm256_setzero_pd();
      __m256d acc02 = _mm256_setzero_pd();
      __m256d acc03 = _mm256_setzero_pd();

      for (int z = 0; z < kMax; z += 4) {
        __m256d a0 = _mm256_loadu_pd(&lhs[i * lhsCols + lhsCol + z]);
        __m256d b0 = _mm256_loadu_pd(&rhsBlockTranspose[y * kMax + z]);
        __m256d b1 = _mm256_loadu_pd(&rhsBlockTranspose[(y + 1) * kMax + z]);
        __m256d b2 = _mm256_loadu_pd(&rhsBlockTranspose[(y + 2) * kMax + z]);
        __m256d b3 = _mm256_loadu_pd(&rhsBlockTranspose[(y + 3) * kMax + z]);

        acc00 = _mm256_fmadd_pd(a0, b0, acc00);
        acc01 = _mm256_fmadd_pd(a0, b1, acc01);
        acc02 = _mm256_fmadd_pd(a0, b2, acc02);
        acc03 = _mm256_fmadd_pd(a0, b3, acc03);
      }

      __m256d hadd01 = _mm256_hadd_pd(acc00, acc01);
      __m256d hadd23 = _mm256_hadd_pd(acc02, acc03);

      __m128d lo01 = _mm256_castpd256_pd128(hadd01);
      __m128d hi01 = _mm256_extractf128_pd(hadd01, 1);
      __m128d sum01 = _mm_add_pd(lo01, hi01);

      __m128d lo23 = _mm256_castpd256_pd128(hadd23);
      __m128d hi23 = _mm256_extractf128_pd(hadd23, 1);
      __m128d sum23 = _mm_add_pd(lo23, hi23);

      __m256d sum = _mm256_set_m128d(sum23, sum01);
      __m256d existing = _mm256_loadu_pd(&result[i * rhsCols + rhsCol + y]);
      _mm256_storeu_pd(&result[i * rhsCols + rhsCol + y],
                       _mm256_add_pd(existing, sum));
    }
  }
}

void matrixMultBlockTransposeVector(const double* __restrict lhs,
                                    const double* __restrict rhs,
                                    double* __restrict result, int lhsRows,
                                    int lhsCols, int rhsCols) {
#pragma omp parallel
  {
    double* rhsBlockTranspose = nullptr;
    posix_memalign(reinterpret_cast<void**>(&rhsBlockTranspose), 64,
                   BIGBLOCKSIZE_K * BIGBLOCKSIZE_J * sizeof(double));

#pragma omp for schedule(dynamic)
    for (int rhsCol = 0; rhsCol <= rhsCols - BIGBLOCKSIZE_J;
         rhsCol += BIGBLOCKSIZE_J) {
      for (int lhsCol = 0; lhsCol <= lhsCols - BIGBLOCKSIZE_K;
           lhsCol += BIGBLOCKSIZE_K) {
        rhsTranspose(rhsBlockTranspose, rhs, lhsCol, rhsCol, rhsCols,
                     BIGBLOCKSIZE_K, BIGBLOCKSIZE_J);

        for (int lhsRow = 0; lhsRow <= lhsRows - BIGBLOCKSIZE_I;
             lhsRow += BIGBLOCKSIZE_I) {
          matrixMultMicroKernel(lhs, rhsBlockTranspose, result, lhsRow, rhsCol,
                                lhsCol, rhsCols, lhsCols, BIGBLOCKSIZE_I,
                                BIGBLOCKSIZE_J, BIGBLOCKSIZE_K);
        }

        const int iTailStart = lhsRows - (lhsRows % BIGBLOCKSIZE_I);
        const int iTailCount = lhsRows - iTailStart;
        if (iTailCount > 0) {
          matrixMultMicroKernelITail(
              lhs, rhsBlockTranspose, result, iTailStart, rhsCol, lhsCol,
              rhsCols, lhsCols, iTailCount, BIGBLOCKSIZE_J, BIGBLOCKSIZE_K);
        }
      }

      for (int lhsCol = lhsCols - (lhsCols % BIGBLOCKSIZE_K); lhsCol < lhsCols;
           lhsCol++) {
        int lhsRow = 0;
        for (; lhsRow <= lhsRows - BIGBLOCKSIZE_I; lhsRow += BIGBLOCKSIZE_I) {
          for (int x = 0; x < BIGBLOCKSIZE_I; x++) {
            for (int y = 0; y < BIGBLOCKSIZE_J; y++) {
              result[(lhsRow + x) * rhsCols + rhsCol + y] +=
                  lhs[(lhsRow + x) * lhsCols + lhsCol] *
                  rhs[lhsCol * rhsCols + rhsCol + y];
            }
          }
        }
        for (; lhsRow < lhsRows; lhsRow++) {
          for (int y = 0; y < BIGBLOCKSIZE_J; y++) {
            result[lhsRow * rhsCols + rhsCol + y] +=
                lhs[lhsRow * lhsCols + lhsCol] *
                rhs[lhsCol * rhsCols + rhsCol + y];
          }
        }
      }
    }
    free(rhsBlockTranspose);
  }

#pragma omp parallel for schedule(dynamic)
  for (int rhsCol = rhsCols - (rhsCols % BIGBLOCKSIZE_J); rhsCol < rhsCols;
       rhsCol++) {
    int lhsCol = 0;
    for (; lhsCol <= lhsCols - BIGBLOCKSIZE_K; lhsCol += BIGBLOCKSIZE_K) {
      int lhsRow = 0;
      for (; lhsRow <= lhsRows - BIGBLOCKSIZE_I; lhsRow += BIGBLOCKSIZE_I) {
        for (int x = 0; x < BIGBLOCKSIZE_I; x++) {
          for (int z = 0; z < BIGBLOCKSIZE_K; z++) {
            result[(lhsRow + x) * rhsCols + rhsCol] +=
                lhs[(lhsRow + x) * lhsCols + lhsCol + z] *
                rhs[(lhsCol + z) * rhsCols + rhsCol];
          }
        }
      }
      for (; lhsRow < lhsRows; lhsRow++) {
        for (int z = 0; z < BIGBLOCKSIZE_K; z++) {
          result[lhsRow * rhsCols + rhsCol] +=
              lhs[lhsRow * lhsCols + lhsCol + z] *
              rhs[(lhsCol + z) * rhsCols + rhsCol];
        }
      }
    }
    for (; lhsCol < lhsCols; lhsCol++) {
      int lhsRow = 0;
      for (; lhsRow <= lhsRows - BIGBLOCKSIZE_I; lhsRow += BIGBLOCKSIZE_I) {
        for (int x = 0; x < BIGBLOCKSIZE_I; x++) {
          result[(lhsRow + x) * rhsCols + rhsCol] +=
              lhs[(lhsRow + x) * lhsCols + lhsCol] *
              rhs[lhsCol * rhsCols + rhsCol];
        }
      }
      for (; lhsRow < lhsRows; lhsRow++) {
        result[lhsRow * rhsCols + rhsCol] +=
            lhs[lhsRow * lhsCols + lhsCol] * rhs[lhsCol * rhsCols + rhsCol];
      }
    }
  }
}
}  // namespace mattTorch::tensor::kernels::cpu
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// bugprone-easily-swappable-parameters)
