#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-6;

// ============ ReLU Layer Tests ============

bool testReLUForward(mattTorch::Tensor& input) {
  mattTorch::ReLU relu;
  mattTorch::Tensor output = relu.forward(input);

  for (int i = 0; i < input.getNValues(); i++) {
    double x = input.getValueDirect(i);
    double expected = x > 0 ? x : 0;
    if (std::abs(output.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testReLUBackward(mattTorch::Tensor& input) {
  mattTorch::ReLU relu;
  mattTorch::Tensor output = relu.forward(input);

  mattTorch::Tensor grad(output.getDimensions());
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

bool testTanhForward(mattTorch::Tensor& input) {
  mattTorch::Tanh tanhLayer;
  mattTorch::Tensor output = tanhLayer.forward(input);

  for (int i = 0; i < input.getNValues(); i++) {
    double expected = std::tanh(input.getValueDirect(i));
    if (std::abs(output.getValueDirect(i) - expected) > TOL) return false;
  }
  return true;
}

bool testTanhBackward(mattTorch::Tensor& input) {
  mattTorch::Tanh tanhLayer;
  mattTorch::Tensor output = tanhLayer.forward(input);

  mattTorch::Tensor grad(output.getDimensions());
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

bool testSoftmaxForward(mattTorch::Tensor& input) {
  mattTorch::Softmax softmax;
  mattTorch::Tensor output = softmax.forward(input);

  mattTorch::Dims dims = input.getDimensions();
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

bool testSoftmaxBackward(mattTorch::Tensor& input) {
  mattTorch::Softmax softmax;
  mattTorch::Tensor output = softmax.forward(input);

  mattTorch::Tensor grad(output.getDimensions());
  grad = 1.0;
  output.backward(grad);

  mattTorch::Dims dims = input.getDimensions();
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
  mattTorch::Tensor input = randomTensor({batchSize, inputs});
  mattTorch::Tensor output = fc.forward(input);

  mattTorch::Dims outDims = output.getDimensions();
  if (outDims[0] != batchSize || outDims[1] != outputs) return false;

  auto params = fc.getParameters();
  mattTorch::Tensor& weight = *params[0];
  mattTorch::Tensor& bias = *params[1];

  for (int b = 0; b < batchSize; b++) {
    for (int o = 0; o < outputs; o++) {
      double expected = bias[mattTorch::Dims{o}];
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
  mattTorch::Tensor input = randomTensor({batchSize, inputs});
  mattTorch::Tensor output = fc.forward(input);

  mattTorch::Tensor grad(output.getDimensions());
  grad = 1.0;
  output.backward(grad);

  auto params = fc.getParameters();
  mattTorch::Tensor& weight = *params[0];
  mattTorch::Tensor& bias = *params[1];

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

int main() {
  mattTorch::Tensor a1 = randomTensor({4, 8});
  mattTorch::Tensor a2 = randomTensor({4, 8});
  mattTorch::Tensor a3 = randomTensor({4, 8});
  mattTorch::Tensor a4 = randomTensor({4, 8});
  mattTorch::Tensor a5 = randomTensor({4, 8});
  mattTorch::Tensor a6 = randomTensor({4, 8});

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
