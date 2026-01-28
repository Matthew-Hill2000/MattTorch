#pragma once

namespace mattTorch::tensor::kernels::cpu {

void matrixMultiplication(const double* __restrict lhs,
                          const double* __restrict rhs,
                          double* __restrict result, int lhsRows, int lhsCols,
                          int rhsCols);

void matrixMultiplicationInverse(const double* __restrict lhs,
                                 const double* __restrict rhs,
                                 double* __restrict result, int lhsRows,
                                 int lhsCols, int rhsCols);

void matrixMultiplicationInverseVector(const double* __restrict lhs,
                                       const double* __restrict rhs,
                                       double* __restrict result, int lhsRows,
                                       int lhsCols, int rhsCols);

void matrixMultiplicationBlock(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, int lhsRows,
                               int lhsCols, int rhsCols);

void matrixMultiplicationBlockVector(const double* __restrict lhs,
                                     const double* __restrict rhs,
                                     double* __restrict result, int lhsRows,
                                     int lhsCols, int rhsCols);
void transposeMultiplicationBlockVectorLHS(const double* __restrict lhs,
                                     const double* __restrict rhs,
                                     double* __restrict result, int lhsRows,
                                     int lhsCols, int rhsCols);
void transposeMultiplicationBlockVectorRHS(const double* __restrict lhs,
                                     const double* __restrict rhs,
                                     double* __restrict result, int lhsRows,
                                     int lhsCols, int rhsCols);
}  // namespace kernels::cpu::matrix
