#include <mattTorch/mattTorch.h>

#include <cmath>
#include <iostream>

constexpr double TOL = 1e-9;

// ============ Tests ============

bool testMSELoss() {
  mattTorch::TensorView input({2, 4});
  mattTorch::TensorView target({2, 4});
  target.setRequiresGrad(false);

  for (int i = 0; i < input.getNValues() / 2; i++) {
    input.getData()[i] = i;
    target.getData()[i] = i + 0.5;
  }
  for (int i = input.getNValues() / 2; i < input.getNValues(); i++) {
    input.getData()[i] = i;
    target.getData()[i] = i + 1.0;
  }

  std::cout << input << std::endl;
  std::cout << target << std::endl;
  mattTorch::criterion::MSELoss mse;
  auto loss = mse.calculateLoss(input, target);

  mattTorch::TensorView gradOutput(loss.getDimensions());
  gradOutput = 1.0;
  loss.backward(gradOutput);

  std::cout << input.detachGradient() << std::endl;

  double* actualGrad = input.detachGradient().getData();
  int n = input.getNValues();

  for (int i = 0; i < n; i++) {
    double expected = (2.0 / n) * (input.getData()[i] - target.getData()[i]);
    if (std::abs(actualGrad[i] - expected) > TOL) return false;
  }
  return true;
}

bool testCrossEntropyLoss() {
  mattTorch::TensorView input({1, 4});
  mattTorch::TensorView target({1, 4});
  target.setRequiresGrad(false);

  for (int i = 0; i < input.getNValues(); i++) {
    input.getData()[i] = 0.1*(i+1);
  }
  target[{0,2}] = 1.0;

  std::cout << input << std::endl;
  std::cout << target << std::endl;

  mattTorch::criterion::CrossEntropyLoss ce;
  auto loss = ce.calculateLoss(input, target);


  mattTorch::TensorView gradOutput(loss.getDimensions());
  gradOutput = 1.0;
  loss.backward(gradOutput);

  std::cout << input.detachGradient() << std::endl;
  double* actualGrad = input.detachGradient().getData();

  for (int i = 0; i < 4; i++) {
    double expected = -target.getData()[i] / input.getData()[i];
    if (std::abs(actualGrad[i] - expected) > TOL) return false;
  }
  return true;
}

// ============ Test Runner ============

void run(const std::string& name, bool passed) {
  std::cout << name << ": " << (passed ? "PASSED" : "FAILED") << std::endl;
}

int main() {
  run("MSE Loss", testMSELoss());
  run("Cross Entropy Loss", testCrossEntropyLoss());
}
