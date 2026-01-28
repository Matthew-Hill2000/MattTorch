#include <mattTorch/mattTorch.h>

#include <chrono>
#include <iostream>
#include <random>

mattTorch::TensorView randomTensor(std::vector<int> dims) {
  static unsigned seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);

  mattTorch::TensorView t(dims);
  t.setRequiresGrad(false);
  double* data = t.getData();
  for (int i = 0; i < t.getNValues(); i++) {
    data[i] = dist(gen);
  }
  return t;
}

// ============ Tests ============

void testAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a + b;
}

void testSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a - b;
}

void testMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a * b;
}

void testDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a / b;
}

void testMatMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a.matrixMultiply(b);
}

void testTransposeMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto result = a.transposeMultiply(b, false);
}

void testEquality(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  bool check = (a == b);
}

// ============ Inplace Tests ============

void testInplaceAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  a += b;
}

void testInplaceSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  a -= b;
}

void testInplaceMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  a *= b;
}

void testInplaceDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  a /= b;
}

// ============ Test Runner ============

template <typename F>
void benchmark(const std::string& name, mattTorch::TensorView& a,
               mattTorch::TensorView& b, F test, int iters = 100) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    test(a, b);
  }
  auto end = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
  std::cout << name << ": " << ms / iters << " ms" << std::endl;
}

int main() {
  auto a = randomTensor({1024, 1024});
  auto b = randomTensor({1024, 1024});

  benchmark("MatMul", a, b, testMatMul);
  // benchmark("TransposeMul", a, b, testTransposeMul);
  // benchmark("Add", a, b, testAdd);
  // benchmark("Mul", a, b, testMul);
  // benchmark("Sub", a, b, testSub);
  // benchmark("Div", a, b, testDiv);
  // benchmark("Equality", a, b, testEquality);
  //
  // benchmark("Inplace Add", a, b, testInplaceAdd);
  // benchmark("Inplace Sub", a, b, testInplaceSub);
  // benchmark("Inplace Mul", a, b, testInplaceMul);
  // benchmark("Inplace Div", a, b, testInplaceDiv);
}
