#include <mattTorch/mattTorch.h>

#include <iostream>

#include "mattTorch/optimiser/sgd/sgd.h"

mattTorch::TensorView oneHotEncode(mattTorch::TensorView& input, int size) {
  mattTorch::TensorView output({input.getDimensions()[0], size});
  output = 0.0;

  for (int i{0}; i < input.getDimensions()[0]; i++) {
    int index = static_cast<int>(input[{i, 0}]);
    output[{i, index}] = 1.0;
  }
  return output;
}

int main() {
  mattTorch::dataset MNIST(64);
  MNIST.loadData(
      "/home/matt/Projects/mattTorch/examples/archive/mnist_train.csv");
  MNIST.shuffle();
  MNIST.printNumber();

  double LearningRate{0.001};
  int NUMEPOCHS{100};
  int numBatches{10};

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(784, 1024)
                               .addTanhLayer()
                               .addFullyConnectedLayer(1024, 1024)
                               .addTanhLayer()
                               .addFullyConnectedLayer(1024, 512)
                               .addTanhLayer()
                               .addFullyConnectedLayer(512, 10)
                               .addSoftmaxLayer()
                               .build();

  // net.loadParameters("/home/matt/Projects/mattTorch/params.csv");

  mattTorch::SGD sgd(net.getParameters(), LearningRate);
  mattTorch::criterion::CrossEntropyLoss crossEntropyLoss;

  for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
    double epoch_loss = 0.0;
    // MNIST.shuffle();

    for (int i = 0; i < numBatches; i++) {
      sgd.zeroGrad();

      std::vector<mattTorch::TensorView> batch = MNIST.getBatch(i);

      mattTorch::TensorView output = net.forward(batch[0]);

      mattTorch::TensorView oneHotTargets = oneHotEncode(batch[1], 10);
      mattTorch::TensorView loss =
          crossEntropyLoss.calculateLoss(output, oneHotTargets);
      loss = loss.mean();
      epoch_loss += loss.getData()[0];

      mattTorch::TensorView gradOutput(loss.getDimensions());
      gradOutput = 1.0;
      loss.backward(gradOutput);

      sgd.updateParameters();
    }

    if (epoch % 1 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << epoch_loss << std::endl;
    }
  }

  // net.saveParameters("/home/matt/Projects/mattTorch/params.csv");
}
