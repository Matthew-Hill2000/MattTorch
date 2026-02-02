#include <immintrin.h>

namespace mattTorch::tensor::kernels::cpu {

/**
 * @brief Adds a scalar to all elements of a tensor using AVX2 SIMD intrinsics
 *
 * Computes (tensor[i] + scalar) for each element of the input tensor and
 * stores the result in the result array. The scalar value is broadcast into
 * an AVX2 register and the main loop processes 4 doubles at a time using
 * AVX2 addition intrinsics, with a scalar fallback loop to handle any
 * remaining elements that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param scalar The scalar value to add to each element
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void tensorScalarAdd(const double* __restrict tensor, const double scalar,
                     double* __restrict result, const int nValues);

/**
 * @brief Subtracts a scalar from all elements of a tensor using AVX2 SIMD
 * intrinsics
 *
 * Computes (tensor[i] - scalar) for each element of the input tensor and
 * stores the result in the result array. The scalar value is broadcast into
 * an AVX2 register and the main loop processes 4 doubles at a time using
 * AVX2 subtraction intrinsics, with a scalar fallback loop to handle any
 * remaining elements that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param scalar The scalar value to subtract from each element
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void tensorScalarSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues);

/**
 * @brief Multiplies all elements of a tensor by a scalar using AVX2 SIMD
 * intrinsics
 *
 * Computes (tensor[i] * scalar) for each element of the input tensor and
 * stores the result in the result array. The scalar value is broadcast into
 * an AVX2 register and the main loop processes 4 doubles at a time using
 * AVX2 multiplication intrinsics, with a scalar fallback loop to handle any
 * remaining elements that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param scalar The scalar value to multiply each element by
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void tensorScalarMultiplication(const double* __restrict tensor,
                                const double scalar, double* __restrict result,
                                const int nValues);

/**
 * @brief Divides all elements of a tensor by a scalar using AVX2 SIMD
 * intrinsics
 *
 * Computes (tensor[i] / scalar) for each element of the input tensor and
 * stores the result in the result array. The scalar value is broadcast into
 * an AVX2 register and the main loop processes 4 doubles at a time using
 * AVX2 division intrinsics, with a scalar fallback loop to handle any
 * remaining elements that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param scalar The scalar value to divide each element by
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void tensorScalarDivision(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues);

/**
 * @brief Subtracts all elements of a tensor from a scalar using AVX2 SIMD
 * intrinsics
 *
 * Computes (scalar - tensor[i]) for each element of the input tensor and
 * stores the result in the result array. The scalar value is broadcast into
 * an AVX2 register and the main loop processes 4 doubles at a time using
 * AVX2 subtraction intrinsics, with a scalar fallback loop to handle any
 * remaining elements that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param scalar The scalar value to subtract each element from
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void scalarTensorSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues);

/**
 * @brief Divides a scalar by all elements of a tensor using AVX2 SIMD
 * intrinsics
 *
 * Computes (scalar / tensor[i]) for each element of the input tensor and
 * stores the result in the result array. The scalar value is broadcast into
 * an AVX2 register and the main loop processes 4 doubles at a time using
 * AVX2 division intrinsics, with a scalar fallback loop to handle any
 * remaining elements that do not fill a complete SIMD register.
 *
 * @param tensor Pointer to the input data
 * @param scalar The scalar value to be divided by each element
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void scalarTensorDivision(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues);

/**
 * @brief Raises all elements of a tensor to an integer power using AVX2 SIMD
 * intrinsics
 *
 * Computes (tensor[i] ^ scalar) for each element of the input tensor and
 * stores the result in the result array. The exponentiation is performed via
 * repeated multiplication. The main loop processes 4 doubles at a time,
 * loading each element into an AVX2 register and repeatedly multiplying it
 * with itself using AVX2 multiplication intrinsics for (scalar - 1)
 * iterations. A scalar fallback loop handles any remaining elements that do
 * not fill a complete SIMD register, performing the same repeated
 * multiplication on individual doubles.
 *
 * @param tensor Pointer to the input data
 * @param scalar The integer exponent to raise each element to
 * @param result Pointer to the output data where the result is stored
 * @param nValues The number of elements to process
 */
void elementwiseExponent(const double* __restrict tensor, const int scalar,
                         double* __restrict result, const int nValues);

}  // namespace mattTorch::tensor::kernels::cpu
