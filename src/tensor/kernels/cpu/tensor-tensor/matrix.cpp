#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>

namespace mattTorch::tensor::kernels::cpu {

void matrixMultiplication(const double* __restrict lhs,
                          const double* __restrict rhs,
                          double* __restrict result, int lhsRows, int lhsCols,
                          int rhsCols) {
  for (int i{0}; i < lhsRows; i++) {
    for (int j{0}; j < rhsCols; j++) {
      for (int k{0}; k < lhsCols; k++) {
        result[rhsCols * i + j] += lhs[lhsCols * i + k] * rhs[rhsCols * k + j];
      }
    }
  }
}

void matrixMultiplicationInverse(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsCols) {
  double* rhsTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsTranspose), 32,
                 lhsCols * rhsCols * sizeof(double));

  for (int i{0}; i < lhsCols; i++) {
    for (int j{0}; j < rhsCols; j++) {
      rhsTranspose[lhsCols * j + i] = rhs[rhsCols * i + j];
    }
  }

  for (int i{0}; i < lhsRows; i++) {
    for (int j{0}; j < rhsCols; j++) {
      for (int k{0}; k < lhsCols; k++) {
        result[rhsCols * i + j] +=
            lhs[lhsCols * i + k] * rhsTranspose[lhsCols * j + k];
      }
    }
  }
}

void matrixMultiplicationInverseVector(const double* __restrict lhs,
                                       const double* __restrict rhs,
                                       double* __restrict result, int lhsRows,
                                       int lhsCols, int rhsCols) {
  double* rhsTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsTranspose), 32,
                 lhsCols * rhsCols * sizeof(double));

  for (int i = 0; i < lhsCols; i++) {
    for (int j = 0; j < rhsCols; j++) {
      rhsTranspose[lhsCols * j + i] = rhs[rhsCols * i + j];
    }
  }

  for (int i = 0; i < lhsRows; i++) {
    for (int j = 0; j < rhsCols; j++) {
      __m256d acc = _mm256_setzero_pd();

      int k = 0;
      for (; k + 3 < lhsCols; k += 4) {
        __m256d a = _mm256_loadu_pd(&lhs[lhsCols * i + k]);
        __m256d b = _mm256_loadu_pd(&rhsTranspose[lhsCols * j + k]);
        acc = _mm256_fmadd_pd(a, b, acc);
      }

      __m128d lo = _mm256_castpd256_pd128(acc);
      __m128d hi = _mm256_extractf128_pd(acc, 1);
      __m128d sum = _mm_add_pd(lo, hi);
      sum = _mm_hadd_pd(sum, sum);

      double result_ij = _mm_cvtsd_f64(sum);

      for (; k < lhsCols; k++) {
        result_ij += lhs[lhsCols * i + k] * rhsTranspose[lhsCols * j + k];
      }

      result[rhsCols * i + j] = result_ij;
    }
  }

  free(rhsTranspose);
}

void matrixMultiplicationBlock(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, int lhsRows,
                               int lhsCols, int rhsCols) {
  int blockSize = 32;

  double* rhsBlockTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsBlockTranspose), 32,
                 blockSize * blockSize * sizeof(double));

  // For each Block of rhs along a Row
  int j{0};
  for (; rhsCols - j >= blockSize; j += blockSize) {
    // For each Block of rhs along a column and lhs across a row
    int k{0};
    for (; lhsCols - k >= blockSize; k += blockSize) {
      // Take the transpose of the block of rhs
      // For each Row of the rhs block
      for (int x{0}; x < blockSize; x++) {
        // For each column of the rhs block
        for (int y{0}; y < blockSize; y++) {
          rhsBlockTranspose[y * blockSize + x] = rhs[(k + x) * rhsCols + j + y];
        }
      }

      // Perform the matrix multiplication of the lhs and rhs blocks
      // For each Block of lhs along a Column
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        // For each row along the lhs block
        for (int x{0}; x < blockSize; x++) {
          // For each column along the rhs block
          for (int y{0}; y < blockSize; y++) {
            // For each column along the lhs block and row along the rhs block
            for (int z{0}; z < blockSize; z++) {
              result[(i + x) * rhsCols + j + y] +=
                  lhs[(i + x) * lhsCols + k + z] *
                  rhsBlockTranspose[y * blockSize + z];
            }
          }
        }
      }

      // When you reach the last entire block down a column of lhs, you then
      // need to multiply the rhs block by the remainder block at the bottom of
      // the lhs column
      // For each row remaining in ths column
      for (; i < lhsRows; i++) {
        // for each column of the rhs block
        for (int y{0}; y < blockSize; y++) {
          // For each column of the lhs remainder block and row of the rhs block
          for (int z{0}; z < blockSize; z++) {
            result[i * rhsCols + j + y] +=
                lhs[i * lhsCols + k + z] * rhsBlockTranspose[y * blockSize + z];
          }
        }
      }
    }

    // When you reach the last entire block down a column of the rhs, you then
    // need to multiply the remainder block at the bottom of the rhs column by
    // the remainder block at the end of every row of the lhs
    // For each column/row reamaining in the lhs/rhs remainder block
    for (; k < lhsCols; k++) {
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
  // When you reach the last entire column of blocks in the rhs, you then need
  // to multiply the column of remainder blocks by each corresponding block
  // across each row of blocks of the lhs
  // For each column of the remaining rhs remainder block
  for (; j < rhsCols; j++) {
    // For each remainder block down the last column of rhs and each block
    // across a row of lhs
    int k{0};
    for (; lhsCols - k >= blockSize; k += blockSize) {
      // For each block of lhs down a column
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        // For each row of the lhs block
        for (int x{0}; x < blockSize; x++) {
          // For each column of the lhs block and each row of the rhs remainder
          // block
          for (int z{0}; z < blockSize; z++) {
            result[(i + x) * rhsCols + j] +=
                lhs[(i + x) * lhsCols + k + z] * rhs[(k + z) * rhsCols + j];
          }
        }
      }
      // when you reach the last entire row of blocks, multiply by the bottom
      // right hand residual block
      for (; i < lhsRows; i++) {
        for (int z{0}; z < blockSize; z++) {
          result[i * rhsCols + j] +=
              lhs[i * lhsCols + k + z] * rhs[(k + z) * rhsCols + j];
        }
      }
    }
    // When you reach the last remainder block of the rhs column of remainder
    // blocks of the rhs, you need to then multiply the small bottom right hand
    // corner remainder block of the rhs by each of the remainder blocks at the
    // end of each row of lhs
    // For each row of the little remainder block and column of the lhs
    // remainder blocks
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
    free(rhsBlockTranspose);
  }
}  // namespace kernels::cpu::matrix
}  // namespace mattTorch::tensor::kernels::cpu
