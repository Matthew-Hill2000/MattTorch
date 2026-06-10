#pragma once

namespace mattTorch::tensor::kernels::cpu {

/**
 * @brief Performs matrix multiplication of two matrices using a naive
 * triple-loop implementation
 *
 * Computes the matrix product C = A * B where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (lhsCols x rhsCols), storing
 * the result in the result array of dimension (lhsRows x rhsCols). The
 * multiplication is performed using a straightforward triple-loop that
 * iterates over the rows of lhs, the columns of rhs, and the shared inner
 * dimension, accumulating the dot product for each element of the result.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order
 * @param rhs Pointer to the right-hand side matrix data in row-major order
 * @param result Pointer to the output matrix data in row-major order, must
 * be zero-initialised before calling
 * @param lhsRows The number of rows in the left-hand side matrix
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of rows in the right-hand side matrix
 * @param rhsCols The number of columns in the right-hand side matrix
 */
void matrixMult(const double* __restrict lhs, const double* __restrict rhs,
                double* __restrict result, int lhsRows, int lhsCols,
                int rhsCols);

void matrixMultBlockPackedVector(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsCols);

/**
 * @brief Performs matrix multiplication of two matrices with an explicit
 * full transpose of the right-hand side matrix
 *
 * Computes the matrix product C = A * B where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (lhsCols x rhsCols), storing
 * the result in the result array of dimension (lhsRows x rhsCols). Before
 * performing the multiplication, the entire rhs matrix is transposed into
 * a separate aligned buffer so that the innermost loop traverses both the
 * lhs row and the transposed rhs row contiguously in memory, improving
 * cache performance over the naive implementation. The multiplication
 * itself is performed using a triple-loop that accumulates the dot product
 * for each element of the result.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order
 * @param rhs Pointer to the right-hand side matrix data in row-major order
 * @param result Pointer to the output matrix data in row-major order, must
 * be zero-initialised before calling
 * @param lhsRows The number of rows in the left-hand side matrix
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of rows in the right-hand side matrix
 * @param rhsCols The number of columns in the right-hand side matrix
 */
void matrixMultTranspose(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, int lhsRows, int lhsCols,
                         int rhsCols);

/**
 * @brief Performs matrix multiplication of two matrices with an explicit
 * full transpose of the right-hand side matrix and AVX2 SIMD intrinsics
 *
 * Computes the matrix product C = A * B where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (lhsCols x rhsCols), storing
 * the result in the result array of dimension (lhsRows x rhsCols). Before
 * performing the multiplication, the entire rhs matrix is transposed into
 * a separate aligned buffer so that the innermost loop traverses both the
 * lhs row and the transposed rhs row contiguously in memory. For each
 * element of the result, the dot product along the shared inner dimension
 * is computed using an AVX2 accumulator that processes 4 doubles at a time
 * via fused multiply-add (FMA) intrinsics. The accumulated partial sums
 * within the SIMD register are then reduced to a single scalar via
 * horizontal addition. A scalar fallback loop handles any remaining
 * elements along the inner dimension that do not fill a complete SIMD
 * register.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order
 * @param rhs Pointer to the right-hand side matrix data in row-major order
 * @param result Pointer to the output matrix data in row-major order
 * @param lhsRows The number of rows in the left-hand side matrix
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of rows in the right-hand side matrix
 * @param rhsCols The number of columns in the right-hand side matrix
 */
void matrixMultTransposeVector(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, int lhsRows,
                               int lhsCols, int rhsCols);

/**
 * @brief Performs matrix multiplication of two matrices using cache blocking
 * with a block-local transpose of the right-hand side
 *
 * Computes the matrix product C = A * B where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (lhsCols x rhsCols), storing
 * the result in the result array of dimension (lhsRows x rhsCols). The
 * multiplication is performed using a blocked algorithm with a block size of
 * 32, in which the matrices are decomposed into square blocks and the
 * multiplication is carried out block by block to maximise cache locality.
 *
 * For each block of the rhs matrix, a transposed copy of the block is
 * constructed into an aligned buffer so that the inner products can be
 * computed by traversing both the lhs and the transposed rhs contiguously
 * in memory. The innermost kernel is a scalar triple-loop that computes
 * the dot product for each element of the result block. After processing
 * all complete blocks, the function handles the remainder elements that do
 * not fit into complete blocks along each dimension using scalar
 * multiplication.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order
 * @param rhs Pointer to the right-hand side matrix data in row-major order
 * @param result Pointer to the output matrix data in row-major order, must
 * be zero-initialised before calling
 * @param lhsRows The number of rows in the left-hand side matrix
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of rows in the right-hand side matrix
 * @param rhsCols The number of columns in the right-hand side matrix
 */
void matrixMultBlockTranspose(const double* __restrict lhs,
                              const double* __restrict rhs,
                              double* __restrict result, int lhsRows,
                              int lhsCols, int rhsCols);
/**
 * @brief Performs matrix multiplication of two matrices using AVX2 SIMD
 * intrinsics with cache blocking and OpenMP parallelisation
 *
 * Computes the matrix product C = A * B where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (lhsCols x rhsCols), storing
 * the result in the result array of dimension (lhsRows x rhsCols). The
 * multiplication is performed using a blocked algorithm with a block size of
 * 64, in which the matrices are decomposed into square blocks and the
 * multiplication is carried out block by block to maximise cache locality.
 *
 * For each block of the rhs matrix, a thread-local transposed copy of the
 * block is constructed so that the inner products can be computed by
 * traversing both the lhs and the transposed rhs contiguously in memory.
 * The inner kernel processes a 2x4 microkernel, computing two rows of the
 * lhs against four columns of the rhs simultaneously. For each microkernel,
 * eight AVX2 accumulators are maintained and updated via fused
 * multiply-add (FMA) intrinsics, processing 4 doubles at a time along the
 * shared inner dimension. The accumulated partial sums within each SIMD
 * register are then reduced to scalar dot products via horizontal addition
 * and added to the existing values in the result matrix.
 *
 * After processing all complete blocks, the function handles the remainder
 * elements that do not fit into complete blocks along each dimension. The
 * remainder along the lhs rows is handled by a single-row microkernel that
 * processes 4 rhs columns at a time. The remainder along the shared inner
 * dimension (lhsCols) is handled by scalar multiplication. The remainder
 * along the rhs columns is handled by a separate OpenMP parallel loop that
 * processes one column at a time using scalar operations.
 *
 * The outermost loop over rhs column blocks is parallelised using OpenMP
 * with dynamic scheduling, and each thread allocates its own aligned
 * transpose buffer to avoid shared memory conflicts between threads.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order
 * @param rhs Pointer to the right-hand side matrix data in row-major order
 * @param result Pointer to the output matrix data in row-major order, must
 * be zero-initialised before calling
 * @param lhsRows The number of rows in the left-hand side matrix
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of rows in the right-hand side matrix
 * @param rhsCols The number of columns in the right-hand side matrix
 */
void matrixMultBlockTransposeVector(const double* __restrict lhs,
                                    const double* __restrict rhs,
                                    double* __restrict result, int lhsRows,
                                    int lhsCols, int rhsCols);

/**
 * @brief Performs matrix multiplication of the transpose of the left-hand
 * side matrix with the right-hand side matrix using AVX2 SIMD intrinsics
 * with cache blocking
 *
 * Computes the matrix product C = A^T * B where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (lhsRows x rhsCols), storing
 * the result in the result array of dimension (lhsCols x rhsCols). The
 * multiplication is performed using a blocked algorithm with a block size of
 * 64, in which the matrices are decomposed into square blocks and the
 * multiplication is carried out block by block to maximise cache locality.
 *
 * For each pair of blocks along the shared inner dimension (the rows of both
 * lhs and rhs), aligned transposed copies of both the lhs block and the rhs
 * block are constructed so that the inner products can be computed by
 * traversing the transposed lhs (giving rows of A^T) and the transposed rhs
 * contiguously in memory. The inner kernel processes a 2x4 microkernel,
 * computing two rows of the transposed lhs against four columns of the rhs
 * simultaneously. For each microkernel, eight AVX2 accumulators are
 * maintained and updated via fused multiply-add (FMA) intrinsics, processing
 * 4 doubles at a time along the shared inner dimension. The accumulated
 * partial sums within each SIMD register are then reduced to scalar dot
 * products via horizontal addition and added to the existing values in the
 * result matrix.
 *
 * After processing all complete blocks, the function handles the remainder
 * elements that do not fit into complete blocks along each dimension. The
 * remainder along the lhs columns (rows of the result) is handled by a
 * single-row microkernel that gathers elements from the lhs column into an
 * AVX2 register and processes 4 rhs columns at a time. The remainder along
 * the shared inner dimension (lhsRows) and the remainder along the rhs
 * columns are both handled by scalar multiplication.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order,
 * which is implicitly transposed during the computation
 * @param rhs Pointer to the right-hand side matrix data in row-major order
 * @param result Pointer to the output matrix data in row-major order, must
 * be zero-initialised before calling
 * @param lhsRows The number of rows in the left-hand side matrix and the
 * number of rows in the right-hand side matrix (the shared inner dimension)
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of rows in the result matrix
 * @param rhsCols The number of columns in the right-hand side matrix and the
 * number of columns in the result matrix
 */
void transposeMultBlockVectorLHS(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsCols);

/**
 * @brief Performs matrix multiplication of the left-hand side matrix with
 * the transpose of the right-hand side matrix using AVX2 SIMD intrinsics
 * with cache blocking
 *
 * Computes the matrix product C = A * B^T where A is of dimension
 * (lhsRows x lhsCols) and B is of dimension (rhsRows x lhsCols), storing
 * the result in the result array of dimension (lhsRows x rhsRows). The
 * multiplication is performed using a blocked algorithm with a block size of
 * 64, in which the matrices are decomposed into square blocks and the
 * multiplication is carried out block by block to maximise cache locality.
 *
 * For each block of the rhs matrix along the row dimension, the block is
 * copied contiguously into an aligned buffer using std::memcpy so that the
 * rows of rhs (which become columns of B^T) can be traversed contiguously
 * in memory during the inner product computation. The lhs rows are already
 * contiguous in row-major order and so do not require transposition. The
 * inner kernel processes a 2x4 microkernel, computing two rows of the lhs
 * against four rows of the rhs (four columns of B^T) simultaneously. For
 * each microkernel, eight AVX2 accumulators are maintained and updated via
 * fused multiply-add (FMA) intrinsics, processing 4 doubles at a time along
 * the shared inner dimension. The accumulated partial sums within each SIMD
 * register are then reduced to scalar dot products via horizontal addition
 * and added to the existing values in the result matrix.
 *
 * After processing all complete blocks, the function handles the remainder
 * elements that do not fit into complete blocks along each dimension. The
 * remainder along the lhs rows (rows of the result) is handled by a
 * single-row microkernel that processes 4 rhs rows at a time. The remainder
 * along the shared inner dimension (lhsCols) and the remainder along the
 * rhs rows (columns of the result) are both handled by scalar
 * multiplication.
 *
 * @param lhs Pointer to the left-hand side matrix data in row-major order
 * @param rhs Pointer to the right-hand side matrix data in row-major order,
 * which is implicitly transposed during the computation
 * @param result Pointer to the output matrix data in row-major order, must
 * be zero-initialised before calling
 * @param lhsRows The number of rows in the left-hand side matrix and the
 * number of rows in the result matrix
 * @param lhsCols The number of columns in the left-hand side matrix and the
 * number of columns in the right-hand side matrix (the shared inner
 * dimension)
 * @param rhsRows The number of rows in the right-hand side matrix and the
 * number of columns in the result matrix
 */
void transposeMultBlockVectorRHS(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsRows);
}  // namespace mattTorch::tensor::kernels::cpu
