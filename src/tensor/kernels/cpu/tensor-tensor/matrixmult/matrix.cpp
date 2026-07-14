// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// bugprone-easily-swappable-parameters)
#include <immintrin.h>
#include <mattTorch/tensor/kernels/cpu/tensor-tensor/matrix.h>
#include <mm_malloc.h>
#include <omp.h>

#include <cstdlib>

namespace mattTorch::tensor::kernels::cpu {

const int CACHE_LINE_SIZE_BYTES = 64;
const int N_DOUBLE_PER_CACHE_LINE = CACHE_LINE_SIZE_BYTES / sizeof(double);
const int REGISTER_SIZE = 32;

const int BLOCK_SIZE = 64;

void matrixMult(const double* __restrict lhs, const double* __restrict rhs,
                double* __restrict result, int lhsRows, int lhsCols,
                int rhsCols) {
  for (int i{0}; i < lhsRows; i++) {
    for (int j{0}; j < rhsCols; j++) {
      for (int k{0}; k < lhsCols; k++) {
        result[(rhsCols * i) + j] +=
            lhs[(lhsCols * i) + k] * rhs[(rhsCols * k) + j];
      }
    }
  }
}

void matrixMultTranspose(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, int lhsRows, int lhsCols,
                         int rhsCols) {
  double* rhsTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsTranspose), REGISTER_SIZE,
                 lhsCols * rhsCols * sizeof(double));

  for (int i{0}; i < lhsCols; i++) {
    for (int j{0}; j < rhsCols; j++) {
      rhsTranspose[(lhsCols * j) + i] = rhs[(rhsCols * i) + j];
    }
  }

  for (int i{0}; i < lhsRows; i++) {
    for (int j{0}; j < rhsCols; j++) {
      for (int k{0}; k < lhsCols; k++) {
        result[(rhsCols * i) + j] +=
            lhs[(lhsCols * i) + k] * rhsTranspose[(lhsCols * j) + k];
      }
    }
  }
}

void matrixMultTransposeVector(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, int lhsRows,
                               int lhsCols, int rhsCols) {
  double* rhsTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsTranspose), REGISTER_SIZE,
                 lhsCols * rhsCols * sizeof(double));

  for (int i = 0; i < lhsCols; i++) {
    for (int j = 0; j < rhsCols; j++) {
      rhsTranspose[(lhsCols * j) + i] = rhs[(rhsCols * i) + j];
    }
  }

  for (int i = 0; i < lhsRows; i++) {
    for (int j = 0; j < rhsCols; j++) {
      __m256d acc = _mm256_setzero_pd();

      int k = 0;
      for (; k + 3 < lhsCols; k += 4) {
        __m256d a = _mm256_loadu_pd(&lhs[(lhsCols * i) + k]);
        __m256d b = _mm256_loadu_pd(&rhsTranspose[(lhsCols * j) + k]);
        acc = _mm256_fmadd_pd(a, b, acc);
      }

      __m128d lo = _mm256_castpd256_pd128(acc);
      __m128d hi = _mm256_extractf128_pd(acc, 1);
      __m128d sum = _mm_add_pd(lo, hi);
      sum = _mm_hadd_pd(sum, sum);

      double result_ij = _mm_cvtsd_f64(sum);

      for (; k < lhsCols; k++) {
        result_ij += lhs[(lhsCols * i) + k] * rhsTranspose[(lhsCols * j) + k];
      }

      result[(rhsCols * i) + j] = result_ij;
    }
  }

  free(rhsTranspose);
}

void matrixMultBlockTranspose(const double* __restrict lhs,
                              const double* __restrict rhs,
                              double* __restrict result, int lhsRows,
                              int lhsCols, int rhsCols) {
  double* rhsBlockTranspose = nullptr;
  posix_memalign(reinterpret_cast<void**>(&rhsBlockTranspose), REGISTER_SIZE,
                 BLOCK_SIZE * BLOCK_SIZE * sizeof(double));

  int j{0};
  for (; rhsCols - j >= BLOCK_SIZE; j += BLOCK_SIZE) {
    int k{0};
    for (; lhsCols - k >= BLOCK_SIZE; k += BLOCK_SIZE) {
      for (int x{0}; x < BLOCK_SIZE; x++) {
        for (int y{0}; y < BLOCK_SIZE; y++) {
          rhsBlockTranspose[(y * BLOCK_SIZE) + x] =
              rhs[((k + x) * rhsCols) + j + y];
        }
      }

      int i{0};
      for (; lhsRows - i >= BLOCK_SIZE; i += BLOCK_SIZE) {
        for (int x{0}; x < BLOCK_SIZE; x++) {
          for (int y{0}; y < BLOCK_SIZE; y++) {
            for (int z{0}; z < BLOCK_SIZE; z++) {
              result[((i + x) * rhsCols) + j + y] +=
                  lhs[((i + x) * lhsCols) + k + z] *
                  rhsBlockTranspose[(y * BLOCK_SIZE) + z];
            }
          }
        }
      }

      for (; i < lhsRows; i++) {
        for (int y{0}; y < BLOCK_SIZE; y++) {
          for (int z{0}; z < BLOCK_SIZE; z++) {
            result[(i * rhsCols) + j + y] +=
                lhs[(i * lhsCols) + k + z] *
                rhsBlockTranspose[(y * BLOCK_SIZE) + z];
          }
        }
      }
    }

    for (; k < lhsCols; k++) {
      int i{0};
      for (; lhsRows - i >= BLOCK_SIZE; i += BLOCK_SIZE) {
        for (int x{0}; x < BLOCK_SIZE; x++) {
          for (int y{0}; y < BLOCK_SIZE; y++) {
            result[((i + x) * rhsCols) + j + y] +=
                lhs[((i + x) * lhsCols) + k] * rhs[(k * rhsCols) + j + y];
          }
        }
      }
      for (; i < lhsRows; i++) {
        for (int y{0}; y < BLOCK_SIZE; y++) {
          result[(i * rhsCols) + j + y] +=
              lhs[(i * lhsCols) + k] * rhs[(k * rhsCols) + j + y];
        }
      }
    }
  }
  for (; j < rhsCols; j++) {
    int k{0};
    for (; lhsCols - k >= BLOCK_SIZE; k += BLOCK_SIZE) {
      int i{0};
      for (; lhsRows - i >= BLOCK_SIZE; i += BLOCK_SIZE) {
        for (int x{0}; x < BLOCK_SIZE; x++) {
          for (int z{0}; z < BLOCK_SIZE; z++) {
            result[((i + x) * rhsCols) + j] +=
                lhs[((i + x) * lhsCols) + k + z] * rhs[((k + z) * rhsCols) + j];
          }
        }
      }
      for (; i < lhsRows; i++) {
        for (int z{0}; z < BLOCK_SIZE; z++) {
          result[(i * rhsCols) + j] +=
              lhs[(i * lhsCols) + k + z] * rhs[((k + z) * rhsCols) + j];
        }
      }
    }

    for (; k < lhsCols; k++) {
      int i{0};
      for (; lhsRows - i >= BLOCK_SIZE; i += BLOCK_SIZE) {
        // For each row of the lhs remainder block
        for (int x{0}; x < BLOCK_SIZE; x++) {
          result[((i + x) * rhsCols) + j] +=
              lhs[((i + x) * lhsCols) + k] * rhs[(k * rhsCols) + j];
        }
      }
      for (; i < lhsRows; i++) {
        result[(i * rhsCols) + j] +=
            lhs[(i * lhsCols) + k] * rhs[(k * rhsCols) + j];
      }
    }
  }

  free(rhsBlockTranspose);
}

}  // namespace mattTorch::tensor::kernels::cpu
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic,
// bugprone-easily-swappable-parameters)
