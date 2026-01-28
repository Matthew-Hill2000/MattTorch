
#include <immintrin.h>
namespace mattTorch::tensor::kernels::cpu {

void ReLU(double* __restrict tensor, double* __restrict result, double* __restrict backwardMask, int nValues);

void broadcast(double* __restrict tensor, double* __restrict result,
               int blockSize, int numBlocks, int repeatCount);

void tanh(double* __restrict tensor, double* __restrict result, int nValues);

void exponential(double* __restrict tensor, double* __restrict result, int nValues);
void log(double* __restrict tensor, double* __restrict result, int nValues);
void mean(double* __restrict tensor, double* __restrict result, int nValues);
}  // namespace mattTorch::tensor::kernels::cpu
