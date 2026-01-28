#include <mattTorch/mattTorch.h>

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

constexpr double TOL = 1e-6;

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

// ============ ReLU Layer Tests ============

bool testReLUForward(mattTorch::TensorView& input) {
  mattTorch::ReLU relu;
  auto output = relu.forward(input);

  for (int i = 0; i < input.getNValues(); i++) {
    double x = input.getValueDirect(i);
    double expected = x > 0 ? x : 0;
    if (std::abs(output.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testReLUBackward(mattTorch::TensorView& input) {
  mattTorch::ReLU relu;
  auto output = relu.forward(input);

  mattTorch::TensorView grad(output.getDimensions());
  grad = 1.0;
  output.backward(grad);

  double* inputGrad = input.detachGradient().getData();
  for (int i = 0; i < input.getNValues(); i++) {
    double expected = input.getValueDirect(i) > 0 ? 1.0 : 0.0;
    if (std::abs(inputGrad[i] - expected) > TOL) return false;
  }
  return true;
}

// ============ Tanh Layer Tests ============

bool testTanhForward(mattTorch::TensorView& input) {
  mattTorch::Tanh tanhLayer;
  auto output = tanhLayer.forward(input);

  for (int i = 0; i < input.getNValues(); i++) {
    double expected = std::tanh(input.getValueDirect(i));
    if (std::abs(output.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testTanhBackward(mattTorch::TensorView& input) {
  mattTorch::Tanh tanhLayer;
  auto output = tanhLayer.forward(input);

  mattTorch::TensorView grad(output.getDimensions());
  grad = 1.0;
  output.backward(grad);

  double* inputGrad = input.detachGradient().getData();
  for (int i = 0; i < input.getNValues(); i++) {
    double tanhVal = std::tanh(input.getValueDirect(i));
    double expected = 1.0 - tanhVal * tanhVal;
    if (std::abs(inputGrad[i] - expected) > TOL) return false;
  }
  return true;
}

// ============ Softmax Layer Tests ============

bool testSoftmaxForward(mattTorch::TensorView& input) {
  mattTorch::Softmax softmax;
  auto output = softmax.forward(input);

  auto dims = input.getDimensions();
  int batchSize = dims[0];
  int features = dims[1];

  for (int b = 0; b < batchSize; b++) {
    double rowSum = 0;
    for (int f = 0; f < features; f++) {
      double val = output[{b, f}];
      if (val < 0) return false;
      rowSum += val;
    }
    if (std::abs(rowSum - 1.0) > TOL) return false;
  }

  // Check against manual calculation
  for (int b = 0; b < batchSize; b++) {
    double maxVal = input[{b, 0}];
    for (int f = 1; f < features; f++) {
      if (input[{b, f}] > maxVal) maxVal = input[{b, f}];
    }

    double sumExp = 0;
    for (int f = 0; f < features; f++) {
      sumExp += std::exp(input[{b, f}]);
    }

    for (int f = 0; f < features; f++) {
      double expected = std::exp(input[{b, f}]) / sumExp;
      if (std::abs(output[{b, f}] - expected) > TOL) return false;
    }
  }
  return true;
}

bool testSoftmaxBackward(mattTorch::TensorView& input) {
  mattTorch::Softmax softmax;
  auto output = softmax.forward(input);

  mattTorch::TensorView grad(output.getDimensions());
  grad = 1.0;
  output.backward(grad);

  auto dims = input.getDimensions();
  int batchSize = dims[0];
  int features = dims[1];

  double* inputGrad = input.detachGradient().getData();
  for (int b = 0; b < batchSize; b++) {
    double rowSum = 0;
    for (int f = 0; f < features; f++) {
      rowSum += inputGrad[b * features + f];
    }
    if (std::abs(rowSum) > TOL) return false;
  }
  return true;
}

// ============ FullyConnected Layer Tests ============

bool testFullyConnectedForward() {
  int batchSize = 4;
  int inputs = 8;
  int outputs = 6;

  mattTorch::FullyConnectedLayer fc(inputs, outputs);
  auto input = randomTensor({batchSize, inputs});
  auto output = fc.forward(input);

  auto outDims = output.getDimensions();
  if (outDims[0] != batchSize || outDims[1] != outputs) return false;

  auto params = fc.getParameters();
  auto& weight = *params[0];
  auto& bias = *params[1];

  for (int b = 0; b < batchSize; b++) {
    for (int o = 0; o < outputs; o++) {
      double expected = bias[std::vector<int>{o}];
      for (int i = 0; i < inputs; i++) {
        expected += input[{b, i}] * weight[{i, o}];
      }
      if (std::abs(output[{b, o}] - expected) > TOL) return false;
    }
  }
  return true;
}

bool testFullyConnectedBackward() {
  int batchSize = 4;
  int inputs = 8;
  int outputs = 6;

  mattTorch::FullyConnectedLayer fc(inputs, outputs);
  auto input = randomTensor({batchSize, inputs});
  auto output = fc.forward(input);

  mattTorch::TensorView grad(output.getDimensions());
  grad = 1.0;
  output.backward(grad);

  auto params = fc.getParameters();
  auto& weight = *params[0];
  auto& bias = *params[1];

  double* inputGrad = input.detachGradient().getData();
  for (int b = 0; b < batchSize; b++) {
    for (int i = 0; i < inputs; i++) {
      double expected = 0;
      for (int o = 0; o < outputs; o++) {
        expected += weight[{i, o}];
      }
      if (std::abs(inputGrad[b * inputs + i] - expected) > TOL) return false;
    }
  }

  double* weightGrad = weight.detachGradient().getData();
  for (int i = 0; i < inputs; i++) {
    for (int o = 0; o < outputs; o++) {
      double expected = 0;
      for (int b = 0; b < batchSize; b++) {
        expected += input[{b, i}];
      }
      if (std::abs(weightGrad[i * outputs + o] - expected) > TOL) return false;
    }
  }

  double* biasGrad = bias.detachGradient().getData();
  for (int o = 0; o < outputs; o++) {
    if (std::abs(biasGrad[o] - batchSize) > TOL) return false;
  }

  return true;
}

bool testFullyConnectedNumParameters() {
  int inputs = 8;
  int outputs = 6;
  mattTorch::FullyConnectedLayer fc(inputs, outputs);

  int expected = inputs * outputs + outputs;  
  return fc.getNumParameters() == expected;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  auto a1 = randomTensor({4, 8});
  auto a2 = randomTensor({4, 8});
  auto a3 = randomTensor({4, 8});
  auto a4 = randomTensor({4, 8});
  auto a5 = randomTensor({4, 8});
  auto a6 = randomTensor({4, 8});

  std::cout << "=== ReLU Layer ===" << std::endl;
  run("Forward", testReLUForward(a1));
  run("Backward", testReLUBackward(a2));

  std::cout << "\n=== Tanh Layer ===" << std::endl;
  run("Forward", testTanhForward(a3));
  run("Backward", testTanhBackward(a4));

  std::cout << "\n=== Softmax Layer ===" << std::endl;
  run("Forward", testSoftmaxForward(a5));
  run("Backward", testSoftmaxBackward(a6));

  std::cout << "\n=== FullyConnected Layer ===" << std::endl;
  run("Forward", testFullyConnectedForward());
  run("Backward", testFullyConnectedBackward());
  run("NumParameters", testFullyConnectedNumParameters());
}
