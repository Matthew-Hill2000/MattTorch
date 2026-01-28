#include <mattTorch/mattTorch.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "mattTorch/optimiser/sgd/sgd.h"

int main() {
  double LearningRate{0.001};
  int NUMEPOCHS{3000};
  int NUMSAMPLES{100};

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(1, 64)
                               .addTanhLayer()
                               .addFullyConnectedLayer(64, 64)
                               .addTanhLayer()
                               .addFullyConnectedLayer(64, 1)
                               .build();

  mattTorch::SGD sgd(net.getParameters(), LearningRate);
  mattTorch::criterion::MSELoss mse;

  // Generate training data as vectors of individual tensors
  std::vector<mattTorch::TensorView> inputs;
  std::vector<mattTorch::TensorView> targets;

  for (int i = 0; i < NUMSAMPLES; i++) {
    double x = -3.14159 + (6.28318 * i / NUMSAMPLES);
    double y = std::sin(x);

    mattTorch::TensorView input({1, 1});
    input.getData()[0] = x;
    inputs.push_back(input);

    mattTorch::TensorView target({1, 1});
    target.setRequiresGrad(false);
    target.getData()[0] = y;
    targets.push_back(target);
  }

  // Training loop
  for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
    double epoch_loss = 0.0;

    for (int i = 0; i < NUMSAMPLES; i++) {
      sgd.zeroGrad();

      mattTorch::TensorView output = net.forward(inputs[i]);
      mattTorch::TensorView loss = mse.calculateLoss(output, targets[i]);
      epoch_loss += loss.getData()[0];

      mattTorch::TensorView gradOutput({1, 1});
      gradOutput = 1.0;
      loss.backward(gradOutput);

      sgd.updateParameters();
    }

    if (epoch % 100 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << epoch_loss / NUMSAMPLES
                << std::endl;
    }
  }

  // Test
  std::cout << "\nTesting:\n";
  std::cout << std::left << std::setw(12) << "x" << std::setw(14) << "predicted"
            << std::setw(14) << "actual" << std::setw(14) << "error" << '\n';

  for (int i = 0; i < 10; i++) {
    double x = -3.14159 + (6.28318 * i / 10);

    mattTorch::TensorView testInput({1, 1});
    testInput.setRequiresGrad(false);
    testInput.getData()[0] = x;

    mattTorch::TensorView pred = net.forward(testInput);
    double predicted = pred.getData()[0];
    double actual = std::sin(x);

    std::cout << std::left << std::setw(12) << x << std::setw(14) << predicted
              << std::setw(14) << actual << std::setw(14)
              << std::abs(predicted - actual) << '\n';
  }
  return 0;
}
