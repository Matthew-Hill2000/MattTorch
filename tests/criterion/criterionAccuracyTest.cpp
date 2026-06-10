#include <mattTorch/mattTorch.h>

#include <cmath>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

// ============ Tests ============

bool testMSELoss() {
  mattTorch::Tensor input({2, 4});
  mattTorch::Tensor target({2, 4});
  target.setRequiresGrad(false);

  for (int i = 0; i < input.getNValues() / 2; i++) {
    input.getData()[i] = i;
    target.getData()[i] = i + 0.5;
  }
  for (int i = input.getNValues() / 2; i < input.getNValues(); i++) {
    input.getData()[i] = i;
    target.getData()[i] = i + 1.0;
  }

  mattTorch::criterion::MSELoss mse;
  mattTorch::Tensor loss = mse.calculateLoss(input, target);

  loss.backward();

  mattTorch::Tensor grad = input.detachGradient();

  double* actualGrad = grad.getData();
  double* actualLoss = loss.getData();
  int n = input.getNValues();
  double expectedLoss{0};

  for (int i = 0; i < n; i++) {
    expectedLoss += std::pow(input.getData()[i] - target.getData()[i], 2);
    double expectedGrad =
        (2.0 / n) * (input.getData()[i] - target.getData()[i]);

    if (std::abs(actualGrad[i] - expectedGrad) > TOL) {
      return false;
    }
  }

  expectedLoss /= n;

  if (std::abs(expectedLoss - actualLoss[0]) > TOL) {
    return false;
  }
  return true;
}

bool testCrossEntropyLoss() {
  mattTorch::Tensor input({1, 4});
  mattTorch::Tensor target({1, 4});
  target.setRequiresGrad(false);

  for (int i = 0; i < input.getNValues(); i++) {
    input.getData()[i] = 0.1 * (i + 1);
  }
  target[{0, 2}] = 1.0;

  mattTorch::criterion::CrossEntropyLoss ce;
  mattTorch::Tensor loss = ce.calculateLoss(input, target);

  mattTorch::Tensor gradOutput(loss.getDimensions());
  loss.backward();

  mattTorch::Tensor grad = input.detachGradient();
  double* actualGrad = grad.getData();
  double* actualLoss = loss.getData();
  double expectedLoss{0};

  for (int i = 0; i < 4; i++) {
    expectedLoss += -target.getData()[i] * std::log(input.getData()[i]);
    double expectedGrad = -target.getData()[i] / input.getData()[i];

    if (std::abs(actualGrad[i] - expectedGrad) > TOL) {
      return false;
    }
  }

  if (std::abs(expectedLoss - actualLoss[0]) > TOL) {
    return false;
  }
  return true;
}

int main() {
  run("MSE Loss", testMSELoss());
  run("Cross Entropy Loss", testCrossEntropyLoss());
}
