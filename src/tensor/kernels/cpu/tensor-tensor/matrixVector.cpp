#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>

namespace mattTorch::tensor::kernels::cpu {

void matrixMultiplicationBlockVector(const double* __restrict lhs,
                                     const double* __restrict rhs,
                                     double* __restrict result, int lhsRows,
                                     int lhsCols, int rhsCols) {
  constexpr int blockSize = 64;

#pragma omp parallel
  {
    double* rhsBlockTranspose = nullptr;
    posix_memalign(reinterpret_cast<void**>(&rhsBlockTranspose), 32,
                   blockSize * blockSize * sizeof(double));
#pragma omp for schedule(dynamic)
    // For each Block of rhs along a Row
    for (int j = 0; j <= rhsCols - blockSize; j += blockSize) {
      // For each Block of rhs along a column and lhs across a row
      for (int k = 0; k <= lhsCols - blockSize; k += blockSize) {
        // Take the transpose of the block of rhs
        // For each Row of the rhs block
        for (int x{0}; x < blockSize; x++) {
          // For each column of the rhs block
          for (int y{0}; y < blockSize; y++) {
            rhsBlockTranspose[y * blockSize + x] =
                rhs[(k + x) * rhsCols + j + y];
          }
        }

        // Perform the matrix multiplication of the lhs and rhs blocks
        // For each Block of lhs along a Column
        for (int i = 0; i <= lhsRows - blockSize; i += blockSize) {
          // For each row along the lhs block, doing 2 rows at a time
          for (int x{0}; x < blockSize; x += 2) {
            // For each column along the rhs block, doing 4 columns at a time
            for (int y{0}; y < blockSize; y += 4) {
              __m256d acc00 = _mm256_setzero_pd();
              __m256d acc01 = _mm256_setzero_pd();
              __m256d acc02 = _mm256_setzero_pd();
              __m256d acc03 = _mm256_setzero_pd();
              __m256d acc10 = _mm256_setzero_pd();
              __m256d acc11 = _mm256_setzero_pd();
              __m256d acc12 = _mm256_setzero_pd();
              __m256d acc13 = _mm256_setzero_pd();

              // For each column along the lhs block and row along the rhs
              // block, doing 4 values down each at a time (vectorised)
              for (int z{0}; z < blockSize; z += 4) {
                // load 4 values along each of the two rows of lhs
                __m256d lhs0 = _mm256_loadu_pd(&lhs[(i + x) * lhsCols + k + z]);
                __m256d lhs1 =
                    _mm256_loadu_pd(&lhs[(i + x + 1) * lhsCols + k + z]);

                // load 4 values down each of the four columns of the rhs, and
                // so equivalently across the four rows of the rhsTransposed
                __m256d rhs0 =
                    _mm256_loadu_pd(&rhsBlockTranspose[y * blockSize + z]);
                __m256d rhs1 = _mm256_loadu_pd(
                    &rhsBlockTranspose[(y + 1) * blockSize + z]);
                __m256d rhs2 = _mm256_loadu_pd(
                    &rhsBlockTranspose[(y + 2) * blockSize + z]);
                __m256d rhs3 = _mm256_loadu_pd(
                    &rhsBlockTranspose[(y + 3) * blockSize + z]);

                // multiply the contents of lhsX and rhsX down each channel and
                // add the contents of accXX down each channel and then assign
                // to accXX. Essentially each channel i of accxy then
                // corresponds to the sum of all multiplications along the 0th
                // lhs and yth rhs in modulo i positions
                acc00 = _mm256_fmadd_pd(lhs0, rhs0, acc00);
                acc01 = _mm256_fmadd_pd(lhs0, rhs1, acc01);
                acc02 = _mm256_fmadd_pd(lhs0, rhs2, acc02);
                acc03 = _mm256_fmadd_pd(lhs0, rhs3, acc03);
                acc10 = _mm256_fmadd_pd(lhs1, rhs0, acc10);
                acc11 = _mm256_fmadd_pd(lhs1, rhs1, acc11);
                acc12 = _mm256_fmadd_pd(lhs1, rhs2, acc12);
                acc13 = _mm256_fmadd_pd(lhs1, rhs3, acc13);
              }

              // Vectorized horizontal sums for row x

              // add the contents of each of the two values within each lane of
              // acc00 and acc01, then store the results in a new __m256d
              // |a00(0)+a00(1) a01(0)+a01(1) | a00(2)+a00(3) a01(2)+a01(3) |
              __m256d hadd01_0 = _mm256_hadd_pd(acc00, acc01);
              // |a02(0)+a02(1) a03(0)+a03(1) | a02(2)+a02(3) a03(2)+a03(3) |
              __m256d hadd23_0 = _mm256_hadd_pd(acc02, acc03);

              // extract the lower lane
              // |a00(0)+a00(1) a01(0)+a01(1)|
              __m128d lo01_0 = _mm256_castpd256_pd128(hadd01_0);
              // extract the higher lane
              // |a00(2)+a00(3) a01(2)+a01(3)|
              __m128d hi01_0 = _mm256_extractf128_pd(hadd01_0, 1);
              // add the contents of the lower lane to the higher lane down each
              // channel
              // |a00(0)+a00(1)+a00(2)+a00(3) a01(0)+a01(1)+a01(2)+a01(3)|
              __m128d sum01_0 = _mm_add_pd(lo01_0, hi01_0);

              // extract the lower lane
              // |a02(0)+a02(1) a03(0)+a03(1)|
              __m128d lo23_0 = _mm256_castpd256_pd128(hadd23_0);
              // extract the higher lane
              // |a02(2)+a02(3) a03(2)+a03(3)|
              __m128d hi23_0 = _mm256_extractf128_pd(hadd23_0, 1);
              // add the contents of the lower lane to the higher lane down each
              // channel
              // |a02(0)+a02(1)+a02(2)+a02(3) a03(0)+a03(1)+a03(2)+a03(3)|
              __m128d sum23_0 = _mm_add_pd(lo23_0, hi23_0);

              // stack the two __m128d sums into a __m256d
              // |a00(0)+a00(1)+a00(2)+a00(3) a01(0)+a01(1)+a01(2)+a01(3)|
              // |a02(0)+a02(1)+a02(2)+a02(3) a03(0)+a03(1)+a03(2)+a03(3)|
              __m256d row0_results = _mm256_set_m128d(sum23_0, sum01_0);

              // load the 4 values of the resulting tensor at the
              // x-y/(y+1)/(y+2)/(y+3) positions
              __m256d existing0 =
                  _mm256_loadu_pd(&result[(i + x) * rhsCols + j + y]);
              // add the values of the matrix-matrix sums across z for each of
              // the 4 rhs columns for the first lhs row
              _mm256_storeu_pd(&result[(i + x) * rhsCols + j + y],
                               _mm256_add_pd(existing0, row0_results));

              // Vectorized horizontal sums for row x+1

              // add the contents of each of the two values within each lane of
              // acc00 and acc01, then store the results in a new __m256d
              __m256d hadd01_1 = _mm256_hadd_pd(acc10, acc11);
              __m256d hadd23_1 = _mm256_hadd_pd(acc12, acc13);

              // extract the lower lane into a __m128d
              __m128d lo01_1 = _mm256_castpd256_pd128(hadd01_1);
              // extract the higher lane into a __m128d
              __m128d hi01_1 = _mm256_extractf128_pd(hadd01_1, 1);
              // add the contents of the lower lane to the higher lane down each
              // channel
              __m128d sum01_1 = _mm_add_pd(lo01_1, hi01_1);

              // extract the lower lane into a __m128d
              __m128d lo23_1 = _mm256_castpd256_pd128(hadd23_1);
              // extract the higher lane into a __m128d
              __m128d hi23_1 = _mm256_extractf128_pd(hadd23_1, 1);
              // add the contents of the lower lane to the higher lane down each
              // channel
              __m128d sum23_1 = _mm_add_pd(lo23_1, hi23_1);

              // stack the two __m128d sums into a __m256d
              __m256d row1_results = _mm256_set_m128d(sum23_1, sum01_1);
              // load the 4 values of the resulting tensor at the
              // (x+1)-y/(y+1)/(y+2)/(y+3) positions
              __m256d existing1 =
                  _mm256_loadu_pd(&result[(i + x + 1) * rhsCols + j + y]);
              // add the values of the matrix-matrix sums across z for each of
              // the 4 rhs columns for the second lhs row
              _mm256_storeu_pd(&result[(i + x + 1) * rhsCols + j + y],
                               _mm256_add_pd(existing1, row1_results));
            }
          }
        }

        // When you reach the last entire block down a column of lhs, you then
        // need to multiply the rhs block by the remainder block at the bottom
        // of the lhs column For each row remaining in ths column
        for (int i = lhsRows - (lhsRows % blockSize); i < lhsRows; i++) {
          // for each column of the rhs block
          for (int y{0}; y < blockSize; y += 4) {
            __m256d acc00 = _mm256_setzero_pd();
            __m256d acc01 = _mm256_setzero_pd();
            __m256d acc02 = _mm256_setzero_pd();
            __m256d acc03 = _mm256_setzero_pd();
            // For each column of the lhs remainder block and row of the rhs
            // block
            for (int z{0}; z < blockSize; z += 4) {
              __m256d a0 = _mm256_loadu_pd(&lhs[i * lhsCols + k + z]);
              __m256d b0 =
                  _mm256_loadu_pd(&rhsBlockTranspose[y * blockSize + z]);
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

      // When you reach the last entire block down a column of the rhs, you then
      // need to multiply the remainder block at the bottom of the rhs column by
      // the remainder block at the end of every row of the lhs
      // For each column/row reamaining in the lhs/rhs remainder block
      for (int k = lhsCols - (lhsCols % blockSize); k < lhsCols; k++) {
        // For each remainder block in the lhs
        int i{0};
        for (; lhsRows - i >= blockSize; i += blockSize) {
          // For each row of the remainder block of the lhs
          for (int x{0}; x < blockSize; x++) {
            // For each column of the remainder block of the rhs
            for (int y{0}; y < blockSize; y++) {
              result[((i + x) * rhsCols) + j + y] +=
                  lhs[(i + x) * lhsCols + k] * rhs[k * rhsCols + j + y];
            }
          }
        }
        for (; i < lhsRows; i++) {
          for (int y{0}; y < blockSize; y++) {
            result[i * rhsCols + j + y] +=
                lhs[i * lhsCols + k] * rhs[k * rhsCols + j + y];
          }
        }
      }
    }
    free(rhsBlockTranspose);
  }

  // When you reach the last entire column of blocks in the rhs, you then need
  // to multiply the column of remainder blocks by each corresponding block
  // across each row of blocks of the lhs
  // For each column of the remaining rhs remainder block
#pragma omp parallel for schedule(dynamic)
  for (int j = rhsCols - (rhsCols % blockSize); j < rhsCols; j++) {
    // For each remainder block down the last column of rhs and each block
    // across a row of lhs
    int k{0};
    for (; lhsCols - k >= blockSize; k += blockSize) {
      // For each block of lhs down a column
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        // For each row of the lhs block
        for (int x{0}; x < blockSize; x++) {
          // For each column of the lhs block and each row of the rhs
          // remainder block
          for (int z{0}; z < blockSize; z++) {
            result[(i + x) * rhsCols + j] +=
                lhs[(i + x) * lhsCols + k + z] * rhs[(k + z) * rhsCols + j];
          }
        }
      }
      // when you reach the last entire row of blocks, multiply the bottom
      // residual block by the rhs residual block right hand residual block
      for (; i < lhsRows; i++) {
        for (int z{0}; z < blockSize; z++) {
          result[i * rhsCols + j] +=
              lhs[i * lhsCols + k + z] * rhs[(k + z) * rhsCols + j];
        }
      }
    }
    // When you reach the last remainder block of the rhs column of remainder
    // blocks of the rhs, you need to then multiply the small bottom right
    // hand corner remainder block of the rhs by each of the remainder blocks
    // at the end of each row of lhs For each row of the little remainder
    // block and column of the lhs remainder blocks
    for (; k < lhsCols; k++) {
      // For each remainder block down the lhs column of remainder blocks.
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        // For each row of the lhs remainder block
        for (int x{0}; x < blockSize; x++) {
          result[(i + x) * rhsCols + j] +=
              lhs[(i + x) * lhsCols + k] * rhs[k * rhsCols + j];
        }
      }
      // When you reach the end of the lhs column of remainders
      // Finally multiply the tiny remainder block of lhs bottom corner and
      // the
      // tiny remainder block of rhs bottom corner
      for (; i < lhsRows; i++) {
        result[i * rhsCols + j] += lhs[i * lhsCols + k] * rhs[k * rhsCols + j];
      }
    }
  }
}
}  // namespace mattTorch::tensor::kernels::cpu
