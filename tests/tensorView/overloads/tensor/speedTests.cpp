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

void testReLU(mattTorch::TensorView& a) { auto result = a.ReLU(); }

void testTanh(mattTorch::TensorView& a) { auto result = a.tanh(); }
void testExp(mattTorch::TensorView& a) { auto result = a.exponential(); }
void testReductionSum(mattTorch::TensorView& a) {
  auto result = a.reductionSum(1);
}

void testBroadcast(mattTorch::TensorView& a) {
  auto result = a.broadcast(0, 10);
}

void testMean(mattTorch::TensorView& a) { auto result = a.mean(); }

void testLog(mattTorch::TensorView& a) { auto result = a.log(); }

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
  benchmark("Tanh", large, testTanh);
  benchmark("Log", large, testLog);
  benchmark("Exp", large, testExp);
  benchmark("ReductionSum", large, testReductionSum);
  benchmark("Mean", large, testMean);
  benchmark("Broadcast", small, testBroadcast);
}
