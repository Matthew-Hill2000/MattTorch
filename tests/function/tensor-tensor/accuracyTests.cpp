#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

// ============ Tests ============

bool testAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a + b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad;
}

bool testSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a - b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad * -1.0;
}

bool testMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a * b;
  c.backward();

  return a.detachGradient() == b && b.detachGradient() == a;
}

bool testDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor c = a / b;
  c.backward();

  mattTorch::Tensor expectedA = 1.0 / b;
  mattTorch::Tensor expectedBDenom = b * b;
  mattTorch::Tensor expectedB = -1.0 * a / expectedBDenom;

  return a.detachGradient() == expectedA && b.detachGradient() == expectedB;
}

// ============ Inplace Tests ============

bool testInplaceAdd(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  a += b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad;
}

bool testInplaceSub(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  a -= b;
  mattTorch::Tensor grad(a.getDimensions());
  grad = 1.0;
  a.backward(grad);

  return a.detachGradient() == grad && b.detachGradient() == grad * -1.0;
}

bool testInplaceMul(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a *= b;
  a.backward();

  return a.detachGradient() == b && b.detachGradient() == aOld;
}

bool testInplaceDiv(mattTorch::Tensor& a, mattTorch::Tensor& b) {
  mattTorch::Tensor aOld = a.deepCopy();
  a /= b;
  a.backward();

  mattTorch::Tensor expectedA = 1.0 / b;
  mattTorch::Tensor expectedBDenom = b * b;
  mattTorch::Tensor expectedB = -1.0 * aOld / expectedBDenom;

  return a.detachGradient() == expectedA && b.detachGradient() == expectedB;
}

int main() {
  mattTorch::Tensor a1 = randomTensor({128, 128});
  mattTorch::Tensor b1 = randomTensor({128, 128});
  mattTorch::Tensor a2 = randomTensor({128, 128});
  mattTorch::Tensor b2 = randomTensor({128, 128});
  mattTorch::Tensor a3 = randomTensor({128, 128});
  mattTorch::Tensor b3 = randomTensor({128, 128});
  mattTorch::Tensor a4 = randomTensor({128, 128});
  mattTorch::Tensor b4 = randomTensor({128, 128});

  run("Add", testAdd(a1, b1));
  run("Sub", testSub(a2, b2));
  run("Mul", testMul(a3, b3));
  run("Div", testDiv(a4, b4));

  mattTorch::Tensor a5 = randomTensor({128, 128});
  mattTorch::Tensor b5 = randomTensor({128, 128});
  mattTorch::Tensor a6 = randomTensor({128, 128});
  mattTorch::Tensor b6 = randomTensor({128, 128});
  mattTorch::Tensor a7 = randomTensor({128, 128});
  mattTorch::Tensor b7 = randomTensor({128, 128});
  mattTorch::Tensor a8 = randomTensor({128, 128});
  mattTorch::Tensor b8 = randomTensor({128, 128});

  run("Inplace Add", testInplaceAdd(a5, b5));
  run("Inplace Sub", testInplaceSub(a6, b6));
  run("Inplace Mul", testInplaceMul(a7, b7));
  run("Inplace Div", testInplaceDiv(a8, b8));
}
