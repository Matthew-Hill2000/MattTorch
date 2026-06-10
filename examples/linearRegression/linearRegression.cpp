#include <mattTorch/mattTorch.h>

#include <cmath>
#include <iomanip>
#include <iostream>

#include "mattTorch/dataset/dataset.h"
#include "mattTorch/optimiser/sgd/sgd.h"
#include "mattTorch/tensor/tensor/tensor.h"

int main() {
  double LearningRate{0.1};
  int NUMEPOCHS{3000};
  int NUMSAMPLES{1000};
  int BATCHSIZE{256};
  int NUMBATCHES = (NUMSAMPLES + (BATCHSIZE - 1)) / BATCHSIZE;

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
  mattTorch::Tensor inputs({NUMSAMPLES, 1});
  mattTorch::Tensor targets({NUMSAMPLES, 1});
  targets.setRequiresGrad(false);

  for (int i = 0; i < NUMSAMPLES; i++) {
    double x = -3.14159 + (6.28318 * i / NUMSAMPLES);
    double y = std::sin(x);

    inputs[{i, 0}] = x;
    targets[{i, 0}] = y;
  }

  mattTorch::dataset data(BATCHSIZE, inputs, targets);

  // Training loop
  for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
    double epoch_loss = 0.0;

    for (int i = 0; i < NUMBATCHES; i++) {
      sgd.zeroGrad();

      std::vector<mattTorch::Tensor> batchData = data.getBatch();
      mattTorch::Tensor output = net.forward(batchData[0]);
      mattTorch::Tensor loss = mse.calculateLoss(output, batchData[1]);
      epoch_loss += loss.getData()[0];

      loss.backward();

      sgd.updateParameters();
    }
    data.shuffle();

    if (epoch % 500 == 0) {
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

    mattTorch::Tensor testInput({1, 1});
    testInput.setRequiresGrad(false);
    testInput.getData()[0] = x;

    mattTorch::Tensor pred = net.forward(testInput);
    double predicted = pred.getData()[0];
    double actual = std::sin(x);

    std::cout << std::left << std::setw(12) << x << std::setw(14) << predicted
              << std::setw(14) << actual << std::setw(14)
              << std::abs(predicted - actual) << '\n';
  }
  return 0;
}
