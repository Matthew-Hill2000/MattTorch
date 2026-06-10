#pragma once

namespace mattTorch::tensor::kernels::cpu {

/**
 * @brief Elementwise addition of two tensors using AVX2 SIMD intrinsics
 *
 * Computes (lhs[i] + rhs[i]) for each pair of elements from the two input
 * tensors and stores the result in the result array. The main loop processes
 * 4 doubles at a time using AVX2 addition intrinsics, with a scalar fallback
 * loop to handle any remaining elements that do not fill a complete SIMD
 * register.
 *
 * @param lhs Pointer to the left-hand side input data
 * @param rhs Pointer to the right-hand side input data
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void elementwiseAdd(const double* __restrict lhs, const double* __restrict rhs,
                    double* __restrict result, int nValues);

/**
 * @brief Elementwise subtraction of two tensors using AVX2 SIMD intrinsics
 *
 * Computes (lhs[i] - rhs[i]) for each pair of elements from the two input
 * tensors and stores the result in the result array. The main loop processes
 * 4 doubles at a time using AVX2 subtraction intrinsics, with a scalar
 * fallback loop to handle any remaining elements that do not fill a complete
 * SIMD register.
 *
 * @param lhs Pointer to the left-hand side input data
 * @param rhs Pointer to the right-hand side input data
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void elementwiseSubtract(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, int nValues);

/**
 * @brief Elementwise multiplication of two tensors using AVX2 SIMD intrinsics
 *
 * Computes (lhs[i] * rhs[i]) for each pair of elements from the two input
 * tensors and stores the result in the result array. The main loop processes
 * 4 doubles at a time using AVX2 multiplication intrinsics, with a scalar
 * fallback loop to handle any remaining elements that do not fill a complete
 * SIMD register.
 *
 * @param lhs Pointer to the left-hand side input data
 * @param rhs Pointer to the right-hand side input data
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void elementwiseMultiplication(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, int nValues);

/**
 * @brief Elementwise division of two tensors using AVX2 SIMD intrinsics
 *
 * Computes (lhs[i] / rhs[i]) for each pair of elements from the two input
 * tensors and stores the result in the result array. The main loop processes
 * 4 doubles at a time using AVX2 division intrinsics, with a scalar fallback
 * loop to handle any remaining elements that do not fill a complete SIMD
 * register.
 *
 * @param lhs Pointer to the left-hand side input data
 * @param rhs Pointer to the right-hand side input data
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void elementwiseDivision(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, int nValues);

/**
 * @brief Computes the sum of elements along a specified dimension via a
 * three-loop reduction
 *
 * Reduces the input tensor along a single dimension by summing all elements
 * along that dimension. The reduction is expressed in terms of three sizes
 * derived from the dimension being reduced: outerSize is the product of all
 * dimensions before the reduction dimension, reduceSize is the size of the
 * dimension being reduced, and innerSize is the product of all dimensions
 * after the reduction dimension. For each combination of outer and inner
 * index, the function accumulates the sum across the reduceSize elements
 * along the reduction dimension and stores the result in the output array.
 *
 * @param input Pointer to the input data
 * @param result Pointer to the output data where the reduced result is stored
 * @param outerSize The product of all dimension sizes before the reduction
 * dimension
 * @param reduceSize The size of the dimension being reduced
 * @param innerSize The product of all dimension sizes after the reduction
 * dimension
 */
void reductionSum(const double* __restrict input, double* __restrict result,
                  int outerSize, int reduceSize, int innerSize);

/**
 * @brief In-place elementwise addition of two tensors using AVX2 SIMD
 * intrinsics
 *
 * Computes (lhs[i] + rhs[i]) for each pair of elements from the two input
 * tensors and stores the result directly back into the lhs array, modifying
 * it in place. The main loop processes 4 doubles at a time using AVX2
 * addition intrinsics with unaligned load and store operations, with a scalar
 * fallback loop to handle any remaining elements that do not fill a complete
 * SIMD register.
 *
 * @param lhs Pointer to the left-hand side data, modified in place with the
 * result
 * @param rhs Pointer to the right-hand side input data
 * @param nValues The number of elements to process
 */
void inplaceElementwiseAdd(double* __restrict lhs, const double* __restrict rhs,
                           int nValues);

}  // namespace mattTorch::tensor::kernels::cpu
