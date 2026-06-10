#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>

namespace mattTorch::tensor::kernels::cpu {

void matrixMult(const double* __restrict lhs, const double* __restrict rhs,
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

void matrixMultTranspose(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, int lhsRows, int lhsCols,
                         int rhsCols) {
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

void matrixMultTransposeVector(const double* __restrict lhs,
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

void matrixMultBlockTranspose(const double* __restrict lhs,
                              const double* __restrict rhs,
                              double* __restrict result, int lhsRows,
                              int lhsCols, int rhsCols) {
  int blockSize = 64;

  double* rhsBlockTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsBlockTranspose), 32,
                 blockSize * blockSize * sizeof(double));

  int j{0};
  for (; rhsCols - j >= blockSize; j += blockSize) {
    int k{0};
    for (; lhsCols - k >= blockSize; k += blockSize) {
      for (int x{0}; x < blockSize; x++) {
        for (int y{0}; y < blockSize; y++) {
          rhsBlockTranspose[y * blockSize + x] = rhs[(k + x) * rhsCols + j + y];
        }
      }

      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        for (int x{0}; x < blockSize; x++) {
          for (int y{0}; y < blockSize; y++) {
            for (int z{0}; z < blockSize; z++) {
              result[(i + x) * rhsCols + j + y] +=
                  lhs[(i + x) * lhsCols + k + z] *
                  rhsBlockTranspose[y * blockSize + z];
            }
          }
        }
      }

      for (; i < lhsRows; i++) {
        for (int y{0}; y < blockSize; y++) {
          for (int z{0}; z < blockSize; z++) {
            result[i * rhsCols + j + y] +=
                lhs[i * lhsCols + k + z] * rhsBlockTranspose[y * blockSize + z];
          }
        }
      }
    }

    for (; k < lhsCols; k++) {
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        for (int x{0}; x < blockSize; x++) {
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
  for (; j < rhsCols; j++) {
    int k{0};
    for (; lhsCols - k >= blockSize; k += blockSize) {
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        for (int x{0}; x < blockSize; x++) {
          for (int z{0}; z < blockSize; z++) {
            result[(i + x) * rhsCols + j] +=
                lhs[(i + x) * lhsCols + k + z] * rhs[(k + z) * rhsCols + j];
          }
        }
      }
      for (; i < lhsRows; i++) {
        for (int z{0}; z < blockSize; z++) {
          result[i * rhsCols + j] +=
              lhs[i * lhsCols + k + z] * rhs[(k + z) * rhsCols + j];
        }
      }
    }

    for (; k < lhsCols; k++) {
      int i{0};
      for (; lhsRows - i >= blockSize; i += blockSize) {
        // For each row of the lhs remainder block
        for (int x{0}; x < blockSize; x++) {
          result[(i + x) * rhsCols + j] +=
              lhs[(i + x) * lhsCols + k] * rhs[k * rhsCols + j];
        }
      }
      for (; i < lhsRows; i++) {
        result[i * rhsCols + j] += lhs[i * lhsCols + k] * rhs[k * rhsCols + j];
      }
    }
  }

  free(rhsBlockTranspose);
}

}  // namespace mattTorch::tensor::kernels::cpu
