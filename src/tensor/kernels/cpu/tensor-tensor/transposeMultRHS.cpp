#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>
#include <cstring>

namespace mattTorch::tensor::kernels::cpu {

void transposeMultiplicationBlockVectorRHS(const double* __restrict lhs,
                                           const double* __restrict rhs,
                                           double* __restrict result,
                                           int lhsRows, int lhsCols,
                                           int rhsRows) {
  constexpr int blockSize = 64;

  double* rhsBlock = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsBlock), 32,
                 blockSize * blockSize * sizeof(double));

  // For each block of columns of result (rows of rhs)
  for (int j = 0; j <= rhsRows - blockSize; j += blockSize) {
    // For each block of inner dimension
    for (int k = 0; k <= lhsCols - blockSize; k += blockSize) {
      for (int y = 0; y < blockSize; y++) {
        std::memcpy(&rhsBlock[y * blockSize], &rhs[(j + y) * lhsCols + k],
                    blockSize * sizeof(double));
      }

      // For each block of rows of result ( rows of lhs)
      for (int i = 0; i <= lhsRows - blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x += 2) {
          for (int y = 0; y < blockSize; y += 4) {
            __m256d acc00 = _mm256_setzero_pd();
            __m256d acc01 = _mm256_setzero_pd();
            __m256d acc02 = _mm256_setzero_pd();
            __m256d acc03 = _mm256_setzero_pd();
            __m256d acc10 = _mm256_setzero_pd();
            __m256d acc11 = _mm256_setzero_pd();
            __m256d acc12 = _mm256_setzero_pd();
            __m256d acc13 = _mm256_setzero_pd();

            for (int z = 0; z < blockSize; z += 4) {
              __m256d lhs0 = _mm256_loadu_pd(&lhs[(i + x) * lhsCols + k + z]);
              __m256d lhs1 =
                  _mm256_loadu_pd(&lhs[(i + x + 1) * lhsCols + k + z]);

              __m256d rhs0 = _mm256_loadu_pd(&rhsBlock[y * blockSize + z]);
              __m256d rhs1 =
                  _mm256_loadu_pd(&rhsBlock[(y + 1) * blockSize + z]);
              __m256d rhs2 =
                  _mm256_loadu_pd(&rhsBlock[(y + 2) * blockSize + z]);
              __m256d rhs3 =
                  _mm256_loadu_pd(&rhsBlock[(y + 3) * blockSize + z]);

              acc00 = _mm256_fmadd_pd(lhs0, rhs0, acc00);
              acc01 = _mm256_fmadd_pd(lhs0, rhs1, acc01);
              acc02 = _mm256_fmadd_pd(lhs0, rhs2, acc02);
              acc03 = _mm256_fmadd_pd(lhs0, rhs3, acc03);
              acc10 = _mm256_fmadd_pd(lhs1, rhs0, acc10);
              acc11 = _mm256_fmadd_pd(lhs1, rhs1, acc11);
              acc12 = _mm256_fmadd_pd(lhs1, rhs2, acc12);
              acc13 = _mm256_fmadd_pd(lhs1, rhs3, acc13);
            }

            // Horizontal sums for row x
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
                _mm256_loadu_pd(&result[(i + x) * rhsRows + j + y]);
            _mm256_storeu_pd(&result[(i + x) * rhsRows + j + y],
                             _mm256_add_pd(existing0, row0_results));

            // Horizontal sums for row x+1
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
                _mm256_loadu_pd(&result[(i + x + 1) * rhsRows + j + y]);
            _mm256_storeu_pd(&result[(i + x + 1) * rhsRows + j + y],
                             _mm256_add_pd(existing1, row1_results));
          }
        }
      }

      // Handle remainder rows of lhs (remainder rows of result)
      for (int i = lhsRows - (lhsRows % blockSize); i < lhsRows; i++) {
        for (int y = 0; y < blockSize; y += 4) {
          __m256d acc00 = _mm256_setzero_pd();
          __m256d acc01 = _mm256_setzero_pd();
          __m256d acc02 = _mm256_setzero_pd();
          __m256d acc03 = _mm256_setzero_pd();

          for (int z = 0; z < blockSize; z += 4) {
            __m256d a0 = _mm256_loadu_pd(&lhs[i * lhsCols + k + z]);
            __m256d b0 = _mm256_loadu_pd(&rhsBlock[y * blockSize + z]);
            __m256d b1 = _mm256_loadu_pd(&rhsBlock[(y + 1) * blockSize + z]);
            __m256d b2 = _mm256_loadu_pd(&rhsBlock[(y + 2) * blockSize + z]);
            __m256d b3 = _mm256_loadu_pd(&rhsBlock[(y + 3) * blockSize + z]);

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
          __m256d existing = _mm256_loadu_pd(&result[i * rhsRows + j + y]);
          _mm256_storeu_pd(&result[i * rhsRows + j + y],
                           _mm256_add_pd(existing, sum));
        }
      }
    }

    // Handle remainder of inner dimension (remaining columns of lhs and rhs)
    for (int k = lhsCols - (lhsCols % blockSize); k < lhsCols; k++) {
      // For each full block of lhs rows
      int i = 0;
      for (; lhsRows - i >= blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x++) {
          for (int y = 0; y < blockSize; y++) {
            result[(i + x) * rhsRows + j + y] +=
                lhs[(i + x) * lhsCols + k] * rhs[(j + y) * lhsCols + k];
          }
        }
      }
      // Remainder rows of lhs
      for (; i < lhsRows; i++) {
        for (int y = 0; y < blockSize; y++) {
          result[i * rhsRows + j + y] +=
              lhs[i * lhsCols + k] * rhs[(j + y) * lhsCols + k];
        }
      }
    }
  }

  // Handle remainder columns of result (remainder rows of rhs)
  for (int j = rhsRows - (rhsRows % blockSize); j < rhsRows; j++) {
    // For each full block of inner dimension
    int k = 0;
    for (; lhsCols - k >= blockSize; k += blockSize) {
      // For each full block of lhs rows
      int i = 0;
      for (; lhsRows - i >= blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x++) {
          for (int z = 0; z < blockSize; z++) {
            result[(i + x) * rhsRows + j] +=
                lhs[(i + x) * lhsCols + k + z] * rhs[j * lhsCols + k + z];
          }
        }
      }
      // Remainder rows of lhs
      for (; i < lhsRows; i++) {
        for (int z = 0; z < blockSize; z++) {
          result[i * rhsRows + j] +=
              lhs[i * lhsCols + k + z] * rhs[j * lhsCols + k + z];
        }
      }
    }
    // Remainder of inner dimension
    for (; k < lhsCols; k++) {
      int i = 0;
      for (; lhsRows - i >= blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x++) {
          result[(i + x) * rhsRows + j] +=
              lhs[(i + x) * lhsCols + k] * rhs[j * lhsCols + k];
        }
      }
      for (; i < lhsRows; i++) {
        result[i * rhsRows + j] += lhs[i * lhsCols + k] * rhs[j * lhsCols + k];
      }
    }
  }

  free(rhsBlock);
}

}  // namespace mattTorch::tensor::kernels::cpu
