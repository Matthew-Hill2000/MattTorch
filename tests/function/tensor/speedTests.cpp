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

void testReLU(mattTorch::TensorView& a) {
  auto result = a.ReLU();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);
}

void testTanh(mattTorch::TensorView& a) {
  auto result = a.tanh();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);
}

void testReductionSum(mattTorch::TensorView& a) {
  auto result = a.reductionSum(0);
  mattTorch::TensorView grad({1});
  grad = 1.0;
  result.backward(grad);
}

void testBroadcast(mattTorch::TensorView& a, int dim) {
  auto result = a.broadcast(0, dim);
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);
}

void testMean(mattTorch::TensorView& a) {
  auto result = a.mean();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);
}

void testExp(mattTorch::TensorView& a) {
  auto result = a.exponential();
  mattTorch::TensorView grad(result.getDimensions());
  grad = 1.0;
  result.backward(grad);
}

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
  auto large = randomTensor({1024, 1024});
  auto small = randomTensor({1024});

  benchmark("ReLU", large, testReLU);
  benchmark("Exp", large, testExp);
  benchmark("Tanh", large, testTanh);
  benchmark("ReductionSum", large, testReductionSum);
  benchmark("Mean", large, testMean);
  benchmark("Broadcast", small, [](auto& a) { testBroadcast(a, 1024); });
}
