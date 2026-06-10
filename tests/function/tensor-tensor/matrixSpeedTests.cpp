#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchMatMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.matrixMultiply(b);
  c.backward();
}

void benchTransposeLHS(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.transposeMultiply(b, true);
  c.backward();
}

void benchTransposeRHS(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.transposeMultiply(b, false);
  c.backward();
}

int main() {
  mattTorch::Tensor a = randomTensor({1024, 1024});
  mattTorch::Tensor b = randomTensor({1024, 1024});

  run("MatMul (fwd+bwd)", [&] { benchMatMul(a, b); }, 50);
  run("Transpose LHS A^T*B (fwd+bwd)", [&] { benchTransposeLHS(a, b); }, 50);
  run("Transpose RHS A*B^T (fwd+bwd)", [&] { benchTransposeRHS(a, b); }, 50);
}
