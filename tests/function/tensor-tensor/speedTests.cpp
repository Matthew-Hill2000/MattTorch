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
  double* data = t.getData();
  for (int i = 0; i < t.getNValues(); i++) {
    data[i] = dist(gen);
  }
  return t;
}

// ============ Tests ============

void testMul(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a * b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);
}

void testDiv(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a / b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);
}

void testAdd(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a + b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);
}

void testSub(mattTorch::TensorView& a, mattTorch::TensorView& b) {
  auto c = a - b;
  mattTorch::TensorView grad(a.getDimensions());
  grad = 1.0;
  c.backward(grad);
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

  benchmark("Mul", a, b, testMul);
  benchmark("Div", a, b, testDiv);
  benchmark("Add", a, b, testAdd);
  benchmark("Sub", a, b, testSub);
}
