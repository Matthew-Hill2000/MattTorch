#pragma once

namespace mattTorch::tensor::kernels::cpu {
void elementwiseAdd(const double* __restrict lhs, const double* __restrict rhs,
                    double* __restrict result, const int nValues);

void elementwiseSubtract(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, const int nValues);

void elementwiseMultiplication(const double* __restrict lhs,
                               const double* __restrict rhs,
                               double* __restrict result, const int nValues);

void elementwiseDivision(const double* __restrict lhs,
                         const double* __restrict rhs,
                         double* __restrict result, const int nValues);

void reductionSum(const double* __restrict input, double* __restrict result,
                  int outerSize, int reduceSize, int innerSize);

void inplaceElementwiseAdd(double* __restrict lhs, const double* __restrict rhs,
                           int nValues);
}  // namespace mattTorch::tensor::kernels::cpu
