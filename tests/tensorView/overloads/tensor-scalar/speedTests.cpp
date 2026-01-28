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

double randomScalar() {
  static unsigned seed =
      std::chrono::system_clock::now().time_since_epoch().count();
  static std::default_random_engine gen(seed);
  std::normal_distribution<double> dist(0.0, 1.0);
  return dist(gen);
}

// ============ Tests ============

void testScalarAdd(mattTorch::TensorView& a, double b) { auto result = a + b; }

void testScalarSub(mattTorch::TensorView& a, double b) { auto result = a - b; }

void testScalarMul(mattTorch::TensorView& a, double b) { auto result = a * b; }

void testScalarDiv(mattTorch::TensorView& a, double b) { auto result = a / b; }

void testExponent(mattTorch::TensorView& a, int b) {
  auto result = a.elementwiseExponent(b);
}

void testScalarAssign(mattTorch::TensorView& a, double b) { a = b; }

// ============ Inplace Tests ============

void testInplaceScalarAdd(mattTorch::TensorView& a, double b) { a += b; }

void testInplaceScalarSub(mattTorch::TensorView& a, double b) { a -= b; }

void testInplaceScalarMul(mattTorch::TensorView& a, double b) { a *= b; }

void testInplaceScalarDiv(mattTorch::TensorView& a, double b) { a /= b; }

// ============ Test Runner ============

template <typename F>
void benchmark(const std::string& name, mattTorch::TensorView& a, F test,
               int iters = 100) {
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < iters; i++) {
    test(a);
  }
  auto end = std::chrono::steady_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start)
                .count();
  std::cout << name << ": " << ms / iters << " ms" << std::endl;
}

int main() {
  auto a = randomTensor({1024, 1024});
  double b = randomScalar();

  benchmark("Scalar Add", a, [b](auto& t) { testScalarAdd(t, b); });
  benchmark("Scalar Sub", a, [b](auto& t) { testScalarSub(t, b); });
  benchmark("Scalar Mul", a, [b](auto& t) { testScalarMul(t, b); });
  benchmark("Scalar Div", a, [b](auto& t) { testScalarDiv(t, b); });
  benchmark("Exponent", a, [](auto& t) { testExponent(t, 3); });
  benchmark("Scalar Assign", a, [b](auto& t) { testScalarAssign(t, b); });

  benchmark("Inplace Scalar Add", a,
            [b](auto& t) { testInplaceScalarAdd(t, b); });
  benchmark("Inplace Scalar Sub", a,
            [b](auto& t) { testInplaceScalarSub(t, b); });
  benchmark("Inplace Scalar Mul", a,
            [b](auto& t) { testInplaceScalarMul(t, b); });
  benchmark("Inplace Scalar Div", a,
            [b](auto& t) { testInplaceScalarDiv(t, b); });
}
