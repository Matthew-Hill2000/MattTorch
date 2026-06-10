#pragma once

#include <mattTorch/mattTorch.h>
#include <mm_malloc.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Random number generator
inline std::default_random_engine& testRng() {
  static std::default_random_engine gen(
      std::chrono::system_clock::now().time_since_epoch().count());
  return gen;
}

// Tensor filled from Normal(0, 1)
inline mattTorch::Tensor randomTensor(mattTorch::Dims dims,
                                      bool requiresGrad = true) {
  std::normal_distribution<double> dist(0.0, 1.0);

  mattTorch::Tensor t(dims);

  t.setRequiresGrad(requiresGrad);

  double* data = t.getData();

  for (int i = 0; i < t.getNValues(); i++) {
    data[i] = dist(testRng());
  }

  return t;
}

inline double randomScalar() {
  std::normal_distribution<double> dist(0.0, 1.0);
  return dist(testRng());
}

// Buffer for test result tensors
inline double* createResultBuffer(int size) {
  double* buffer = nullptr;
  posix_memalign(reinterpret_cast<void**>(&buffer), 64, size * sizeof(double));
  memset(buffer, 0, size * sizeof(double));
  return buffer;
}

// ============ Accuracy ============

inline void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

// ============ Speed (median / best) ============

// print the median and best of a vector of times.
inline void printTimes(const std::string& name, std::vector<double>& times) {
  std::sort(times.begin(), times.end());
  const double median = times[times.size() / 2];
  const double best = times.front();
  std::cout << name << ": median " << median << " ms, best " << best << " ms"
            << std::endl;
}

template <typename F>
void run(const std::string& name, F test, int iters = 100) {
  // Warmup pass that pays page-fault, thread-spawn, lazy-init costs once. Not
  // timed.
  test();

  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; i++) {
    auto t0 = std::chrono::steady_clock::now();
    test();
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  printTimes(name, times);
}

template <typename F>
void runBuffer(const std::string& name, F test, int nValues, int iters = 100) {
  double* resultBuffer = createResultBuffer(nValues);
  // Warmup pass that pays page-fault, thread-spawn, lazy-init costs once. Not
  // timed.
  test(resultBuffer);

  std::vector<double> times;
  times.reserve(iters);
  for (int i = 0; i < iters; i++) {
    auto t0 = std::chrono::steady_clock::now();
    test(resultBuffer);
    auto t1 = std::chrono::steady_clock::now();
    times.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  printTimes(name, times);
  free(resultBuffer);
}
