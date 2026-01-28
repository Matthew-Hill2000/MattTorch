#include <immintrin.h>
namespace mattTorch::tensor::kernels::cpu {
void tensorScalarAdd(const double* __restrict tensor, const double scalar,
                     double* __restrict result, const int nValues);

void tensorScalarSubtract(const double* __restrict tensor, const double scalar,
                          double* __restrict result, const int nValues);

void tensorScalarMultiplication(const double* __restrict tensor,
                                const double scalar, double* __restrict result,
                                const int nValues);

void tensorScalarDivision(const double* __restrict tensor, const double scalar,
                           double* __restrict result, const int nValues);

void scalarTensorSubtract(const double* __restrict tensor, const double scalar, double* __restrict result, const int nValues);
void scalarTensorDivision(const double* __restrict tensor, const double scalar, double* __restrict result, const int nValues);

void elementwiseExponent(const double* __restrict tensor, const int scalar,
                         double* __restrict result, const int nValues);

}  // namespace mattTorch::tensor::kernels::cpu
