#include <immintrin.h>

namespace mattTorch::tensor::kernels::cpu {

/**
 * @brief Applies the Rectified Linear Unit function elementwise using AVX2
 * SIMD intrinsics
 *
 * Computes the ReLU function (max(0, x)) for each element of the input tensor
 * and stores the result in the result array. Simultaneously constructs a
 * backwards mask tensor that records which elements were positive (1.0) and
 * which were zeroed out (0.0), for use during the backward pass in gradient
 * computation. The main loop processes 4 doubles at a time using AVX2
 * intrinsics, with a scalar fallback loop to handle any remaining elements
 * that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param result Pointer to the output data where the ReLU result is stored
 * @param backwardMask Pointer to the output data where the backwards mask is
 * stored, with 1.0 for positive elements and 0.0 for zeroed elements
 * @param nValues The number of elements to process
 */
void ReLU(double* __restrict tensor, double* __restrict result,
          double* __restrict backwardMask, int nValues);

/**
 * @brief Broadcasts data by repeating blocks along a new dimension
 *
 * Copies each block of data from the input tensor into the result array,
 * repeating each block the specified number of times. The input tensor is
 * treated as a series of contiguous blocks of size blockSize, and each block
 * is copied repeatCount times into the result array using std::memcpy.
 *
 * @param tensor Pointer to the input data
 * @param result Pointer to the output data where the broadcast result is stored
 * @param blockSize The number of elements in each block to be repeated
 * @param numBlocks The number of blocks in the input tensor
 * @param repeatCount The number of times each block is repeated in the output
 */
void broadcast(double* __restrict tensor, double* __restrict result,
               int blockSize, int numBlocks, int repeatCount);

/**
 * @brief Applies the hyperbolic tangent function elementwise using AVX2 SIMD
 * intrinsics via the SLEEF library
 *
 * Computes the hyperbolic tangent for each element of the input tensor and
 * stores the result in the result array. The main loop processes 4 doubles at
 * a time using the SLEEF Sleef_tanhd4_u35avx2 function, with a scalar
 * fallback loop using std::tanh to handle any remaining elements that do not
 * fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param result Pointer to the output data where the tanh result is stored
 * @param nValues The number of elements to process
 */
void tanh(double* __restrict tensor, double* __restrict result, int nValues);

/**
 * @brief Applies the exponential function elementwise using AVX2 SIMD
 * intrinsics via the SLEEF library
 *
 * Computes the exponential function for each element of the input tensor and
 * stores the result in the result array. The main loop processes 4 doubles at
 * a time using the SLEEF Sleef_expd4_u10avx2 function, with a scalar fallback
 * loop using std::exp to handle any remaining elements that do not fill a
 * complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param result Pointer to the output data where the exponential result is
 * stored
 * @param nValues The number of elements to process
 */
void exponential(double* __restrict tensor, double* __restrict result,
                 int nValues);

/**
 * @brief Applies the natural logarithm function elementwise using AVX2 SIMD
 * intrinsics via the SLEEF library
 *
 * Computes the natural logarithm for each element of the input tensor and
 * stores the result in the result array. The main loop processes 4 doubles at
 * a time using the SLEEF Sleef_logd4_u35avx2 function, with a scalar fallback
 * loop using std::log to handle any remaining elements that do not fill a
 * complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param result Pointer to the output data where the natural logarithm result
 * is stored
 * @param nValues The number of elements to process
 */
void log(double* __restrict tensor, double* __restrict result, int nValues);

/**
 * @brief Computes the arithmetic mean of all elements using AVX2 SIMD
 * intrinsics
 *
 * Computes the arithmetic mean of all elements in the input tensor and stores
 * the scalar result in the result pointer. The main loop accumulates the sum
 * by processing 4 doubles at a time using AVX2 intrinsics, after which the
 * four partial sums within the SIMD register are reduced to a single scalar
 * sum via horizontal addition. A scalar fallback loop accumulates any
 * remaining elements that do not fill a complete SIMD register, before the
 * total sum is divided by nValues to produce the mean.
 *
 * @param tensor Pointer to the input data
 * @param result Pointer to the output scalar where the mean is stored
 * @param nValues The number of elements to process
 */
void mean(double* __restrict tensor, double* __restrict result, int nValues);

}  // namespace mattTorch::tensor::kernels::cpu
