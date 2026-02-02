
#include <mattTorch/mattTorch.h>

#include <iostream>

#include "mattTorch/optimiser/sgd/sgd.h"

int main() {
  double LEARNINGRATE{0.001};
  int NUMEPOCHS{3000};
  int NUMSAMPLES{100};
  double INF{10.0};

  mattTorch::Network net = mattTorch::NetworkBuilder()
                               .addFullyConnectedLayer(1, 128)
                               .addTanhLayer()
                               .addFullyConnectedLayer(128, 128)
                               .addTanhLayer()
                               .addFullyConnectedLayer(128, 1)
                               .build();

  mattTorch::SGD sgd(net.getParameters(), LEARNINGRATE);
  mattTorch::criterion::MSELoss mse;

  mattTorch::TensorView domain(std::vector<int>{NUMSAMPLES, 1});
  mattTorch::TensorView zeroBoundary(std::vector<int>{1, 1});
  mattTorch::TensorView infinityBoundary(std::vector<int>{1, 1});

  for (int i = 0; i < NUMSAMPLES; i++) {
    domain.setValueDirect(i, static_cast<double>(i) * 10.0 / NUMSAMPLES);
  }

  zeroBoundary = 0.0;
  infinityBoundary = 10.0;

  // Training loop
  for (int epoch = 0; epoch < NUMEPOCHS; epoch++) {
    mattTorch::TensorView zeroBoundaryOut = net.forward(zeroBoundary);

    zeroBoundaryOut.backward();
    mattTorch::TensorView zeroBoundaryOutGrad = zeroBoundary.detachGradient();

    mattTorch::TensorView zeroLossOne =
        (zeroBoundaryOut.reductionSum(1)).elementwiseExponent(2);
    mattTorch::TensorView zeroLossTwo =
        (zeroBoundaryOutGrad.reductionSum(1)).elementwiseExponent(2);

    mattTorch::TensorView infinityBoundaryOut = net.forward(infinityBoundary);

    infinityBoundaryOut.backward();
    mattTorch::TensorView infinityBoundaryOutGrad =
        infinityBoundary.detachGradient();

    mattTorch::TensorView infinityLoss =
        (infinityBoundaryOutGrad.reductionSum(1) - 1).elementwiseExponent(2);

    mattTorch::TensorView domainOut = net.forward(domain);

    domainOut.backward(true);
    mattTorch::TensorView domainOutGrad1 = domain.detachGradient();

    domainOutGrad1.backward(true);
    mattTorch::TensorView domainOutGrad2 = domain.detachGradient();

    domainOutGrad2.backward(true);
    mattTorch::TensorView domainOutGrad3 = domain.detachGradient();

    mattTorch::TensorView domainLoss =
        ((domainOutGrad3 + 0.5 * domainOut * domainOutGrad2)
             .elementwiseExponent(2))
            .mean();

    mattTorch::TensorView totalLoss =
        domainLoss + infinityLoss + zeroLossOne + zeroLossTwo;

    sgd.zeroGrad();
    totalLoss.backward();
    sgd.updateParameters();

    if (epoch % 1 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << totalLoss.getValueDirect(0)
                << std::endl;
    }
  }
}
