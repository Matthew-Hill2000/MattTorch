#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

bool testMatMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.matrixMultiply(b);
  mattTorch::Tensor grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::Tensor aGrad(a.getDimensions());
  mattTorch::Tensor bGrad(b.getDimensions());

  int I = a.getDimensions()[0], K = a.getDimensions()[1],
      J = b.getDimensions()[1];

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++)
      for (int j = 0; j < J; j++) {
        aGrad[{i, k}] += grad[{i, j}] * b[{k, j}];
      }

  for (int k = 0; k < K; k++)
    for (int j = 0; j < J; j++)
      for (int i = 0; i < I; i++) {
        bGrad[{k, j}] += a[{i, k}] * grad[{i, j}];
      }

  return a.detachGradient() == aGrad && b.detachGradient() == bGrad;
}

bool testTransposeLHS(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.transposeMultiply(b, true);
  mattTorch::Tensor grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::Tensor aGrad(a.getDimensions());
  mattTorch::Tensor bGrad(b.getDimensions());

  int I = a.getDimensions()[0], J = a.getDimensions()[1],
      K = b.getDimensions()[1];

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++) {
        aGrad[{i, j}] += grad[{j, k}] * b[{i, k}];
      }

  for (int i = 0; i < I; i++)
    for (int k = 0; k < K; k++)
      for (int j = 0; j < J; j++) {
        bGrad[{i, k}] += a[{i, j}] * grad[{j, k}];
      }

  return a.detachGradient() == aGrad && b.detachGradient() == bGrad;
}

bool testTransposeRHS(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a.transposeMultiply(b, false);
  mattTorch::Tensor grad(c.getDimensions());
  grad = 1.0;
  c.backward(grad);

  mattTorch::Tensor aGrad(a.getDimensions());
  mattTorch::Tensor bGrad(b.getDimensions());

  int I = a.getDimensions()[0], J = a.getDimensions()[1],
      K = b.getDimensions()[0];

  for (int i = 0; i < I; i++)
    for (int j = 0; j < J; j++)
      for (int k = 0; k < K; k++) {
        aGrad[{i, j}] += grad[{i, k}] * b[{k, j}];
      }

  for (int k = 0; k < K; k++)
    for (int j = 0; j < J; j++)
      for (int i = 0; i < I; i++) {
        bGrad[{k, j}] += grad[{i, k}] * a[{i, j}];
      }

  return a.detachGradient() == aGrad && b.detachGradient() == bGrad;
}

int main() {
  mattTorch::Tensor a1 = randomTensor({128, 128});
  mattTorch::Tensor b1 = randomTensor({128, 128});

  mattTorch::Tensor a2 = randomTensor({128, 64});
  mattTorch::Tensor b2 = randomTensor({128, 96});

  mattTorch::Tensor a3 = randomTensor({128, 64});
  mattTorch::Tensor b3 = randomTensor({96, 64});

  run("MatMul", testMatMul(a1, b1));
  run("Transpose LHS (A^T * B)", testTransposeLHS(a2, b2));
  run("Transpose RHS (A * B^T)", testTransposeRHS(a3, b3));
}
