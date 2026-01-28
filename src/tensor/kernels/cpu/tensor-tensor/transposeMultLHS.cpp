
#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>

namespace mattTorch::tensor::kernels::cpu {

void transposeMultiplicationBlockVectorLHS(const double* __restrict lhs,
                                           const double* __restrict rhs,
                                           double* __restrict result,
                                           int lhsRows, int lhsCols,
                                           int rhsCols) {
  constexpr int blockSize = 64;

  double* lhsBlockTranspose = nullptr;
  double* rhsBlockTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&lhsBlockTranspose), 32,
                 blockSize * blockSize * sizeof(double));
  posix_memalign(reinterpret_cast<void**>(&rhsBlockTranspose), 32,
                 blockSize * blockSize * sizeof(double));

  // For each block of rhs along a row (columns of result)
  for (int j = 0; j <= rhsCols - blockSize; j += blockSize) {
    // For each block along the inner dimension (rows of lhs and rhs)
    for (int k = 0; k <= lhsRows - blockSize; k += blockSize) {
      // Transpose the block of rhs: rhs[k:k+block, j:j+block]
      for (int x = 0; x < blockSize; x++) {
        for (int y = 0; y < blockSize; y++) {
          rhsBlockTranspose[y * blockSize + x] = rhs[(k + x) * rhsCols + j + y];
        }
      }

      // For each block of lhs columns (rows of result)
      for (int i = 0; i <= lhsCols - blockSize; i += blockSize) {
        // Transpose the block of lhs: lhs[k:k+block, i:i+block]
        // This gives us rows of lhs^T
        for (int x = 0; x < blockSize; x++) {
          for (int y = 0; y < blockSize; y++) {
            lhsBlockTranspose[y * blockSize + x] =
                lhs[(k + x) * lhsCols + i + y];
          }
        }

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
              __m256d lhs0 =
                  _mm256_loadu_pd(&lhsBlockTranspose[x * blockSize + z]);
              __m256d lhs1 =
                  _mm256_loadu_pd(&lhsBlockTranspose[(x + 1) * blockSize + z]);

              __m256d rhs0 =
                  _mm256_loadu_pd(&rhsBlockTranspose[y * blockSize + z]);
              __m256d rhs1 =
                  _mm256_loadu_pd(&rhsBlockTranspose[(y + 1) * blockSize + z]);
              __m256d rhs2 =
                  _mm256_loadu_pd(&rhsBlockTranspose[(y + 2) * blockSize + z]);
              __m256d rhs3 =
                  _mm256_loadu_pd(&rhsBlockTranspose[(y + 3) * blockSize + z]);

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
                _mm256_loadu_pd(&result[(i + x) * rhsCols + j + y]);
            _mm256_storeu_pd(&result[(i + x) * rhsCols + j + y],
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
                _mm256_loadu_pd(&result[(i + x + 1) * rhsCols + j + y]);
            _mm256_storeu_pd(&result[(i + x + 1) * rhsCols + j + y],
                             _mm256_add_pd(existing1, row1_results));
          }
        }
      }

      // Handle remainder columns of lhs (remainder rows of result)
      for (int i = lhsCols - (lhsCols % blockSize); i < lhsCols; i++) {
        for (int y = 0; y < blockSize; y += 4) {
          __m256d acc00 = _mm256_setzero_pd();
          __m256d acc01 = _mm256_setzero_pd();
          __m256d acc02 = _mm256_setzero_pd();
          __m256d acc03 = _mm256_setzero_pd();

          for (int z = 0; z < blockSize; z += 4) {
            // Load 4 values from column i of lhs, rows k+z to k+z+3
            __m256d a0 = _mm256_set_pd(
                lhs[(k + z + 3) * lhsCols + i], lhs[(k + z + 2) * lhsCols + i],
                lhs[(k + z + 1) * lhsCols + i], lhs[(k + z) * lhsCols + i]);
            __m256d b0 = _mm256_loadu_pd(&rhsBlockTranspose[y * blockSize + z]);
            __m256d b1 =
                _mm256_loadu_pd(&rhsBlockTranspose[(y + 1) * blockSize + z]);
            __m256d b2 =
                _mm256_loadu_pd(&rhsBlockTranspose[(y + 2) * blockSize + z]);
            __m256d b3 =
                _mm256_loadu_pd(&rhsBlockTranspose[(y + 3) * blockSize + z]);

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
          __m256d existing = _mm256_loadu_pd(&result[i * rhsCols + j + y]);
          _mm256_storeu_pd(&result[i * rhsCols + j + y],
                           _mm256_add_pd(existing, sum));
        }
      }
    }

    // Handle remainder of inner dimension (remaining rows of lhs and rhs)
    for (int k = lhsRows - (lhsRows % blockSize); k < lhsRows; k++) {
      // For each full block of lhs columns
      int i = 0;
      for (; lhsCols - i >= blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x++) {
          for (int y = 0; y < blockSize; y++) {
            result[(i + x) * rhsCols + j + y] +=
                lhs[k * lhsCols + i + x] * rhs[k * rhsCols + j + y];
          }
        }
      }
      // Remainder columns of lhs
      for (; i < lhsCols; i++) {
        for (int y = 0; y < blockSize; y++) {
          result[i * rhsCols + j + y] +=
              lhs[k * lhsCols + i] * rhs[k * rhsCols + j + y];
        }
      }
    }
  }

  // Handle remainder columns of rhs (remainder columns of result)
  for (int j = rhsCols - (rhsCols % blockSize); j < rhsCols; j++) {
    // For each full block of inner dimension
    int k = 0;
    for (; lhsRows - k >= blockSize; k += blockSize) {
      // For each full block of lhs columns
      int i = 0;
      for (; lhsCols - i >= blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x++) {
          for (int z = 0; z < blockSize; z++) {
            result[(i + x) * rhsCols + j] +=
                lhs[(k + z) * lhsCols + i + x] * rhs[(k + z) * rhsCols + j];
          }
        }
      }
      // Remainder columns of lhs
      for (; i < lhsCols; i++) {
        for (int z = 0; z < blockSize; z++) {
          result[i * rhsCols + j] +=
              lhs[(k + z) * lhsCols + i] * rhs[(k + z) * rhsCols + j];
        }
      }
    }
    // Remainder of inner dimension
    for (; k < lhsRows; k++) {
      int i = 0;
      for (; lhsCols - i >= blockSize; i += blockSize) {
        for (int x = 0; x < blockSize; x++) {
          result[(i + x) * rhsCols + j] +=
              lhs[k * lhsCols + i + x] * rhs[k * rhsCols + j];
        }
      }
      for (; i < lhsCols; i++) {
        result[i * rhsCols + j] += lhs[k * lhsCols + i] * rhs[k * rhsCols + j];
      }
    }
  }

  free(lhsBlockTranspose);
  free(rhsBlockTranspose);
}


}  // namespace mattTorch::tensor::kernels::cpu
