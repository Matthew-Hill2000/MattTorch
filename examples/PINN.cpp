
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

    mattTorch::TensorView ones1(zeroBoundaryOut.getDimensions());
    ones1 = 1.0;
    zeroBoundaryOut.backward(ones1);
    mattTorch::TensorView zeroBoundaryOutGrad = zeroBoundary.detachGradient();

    mattTorch::TensorView zeroLossOne =
        (zeroBoundaryOut.reductionSum(1)).elementwiseExponent(2);
    mattTorch::TensorView zeroLossTwo =
        (zeroBoundaryOutGrad.reductionSum(1)).elementwiseExponent(2);

    mattTorch::TensorView infinityBoundaryOut = net.forward(infinityBoundary);

    mattTorch::TensorView ones2(infinityBoundaryOut.getDimensions());
    ones2 = 1.0;
    infinityBoundaryOut.backward(ones2);
    mattTorch::TensorView infinityBoundaryOutGrad =
        infinityBoundary.detachGradient();

    mattTorch::TensorView infinityLoss =
        (infinityBoundaryOutGrad.reductionSum(1) - 1).elementwiseExponent(2);

    mattTorch::TensorView domainOut = net.forward(domain);

    mattTorch::TensorView ones3(domainOut.getDimensions());
    ones3 = 1.0;
    domainOut.backward(ones3, true);
    mattTorch::TensorView domainOutGrad1 = domain.detachGradient();

    mattTorch::TensorView ones4(domainOut.getDimensions());
    ones4 = 1.0;
    domainOutGrad1.backward(ones4, true);
    mattTorch::TensorView domainOutGrad2 = domain.detachGradient();

    mattTorch::TensorView ones5(domainOut.getDimensions());
    ones5 = 1.0;
    domainOutGrad2.backward(ones5, true);
    mattTorch::TensorView domainOutGrad3 = domain.detachGradient();

    mattTorch::TensorView domainLoss =
        ((domainOutGrad3 + 0.5 * domainOut * domainOutGrad2)
             .elementwiseExponent(2))
            .mean();

    mattTorch::TensorView totalLoss =
        domainLoss + infinityLoss + zeroLossOne + zeroLossTwo;

    mattTorch::TensorView one({1});
    one = 1.0;
    sgd.zeroGrad();
    totalLoss.backward(one);
    sgd.updateParameters();

    if (epoch % 1 == 0) {
      std::cout << "Epoch " << epoch << " Loss: " << totalLoss.getValueDirect(0)
                << std::endl;
    }
  }
}
