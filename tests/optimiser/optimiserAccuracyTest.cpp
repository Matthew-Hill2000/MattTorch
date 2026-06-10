#include <mattTorch/mattTorch.h>
#include <mattTorch/network/network.h>
#include <mattTorch/optimiser/sgd/sgd.h>

#include <cmath>
#include <vector>

#include "common/testUtils.h"

constexpr double TOL = 1e-9;

mattTorch::Network buildNet() {
  return mattTorch::NetworkBuilder().addFullyConnectedLayer(4, 4).build();
}

void populateGradients(mattTorch::Network& net) {
  mattTorch::Tensor input = randomTensor({2, 4}, false);
  mattTorch::Tensor out = net.forward(input);
  out.backward();
}

bool testSGDStep() {
  auto net = buildNet();
  auto params = net.getParameters();
  const double lr = 0.1;
  mattTorch::SGD sgd(params, lr);

  populateGradients(net);

  std::vector<std::vector<double>> before(params.size());
  std::vector<std::vector<double>> grads(params.size());
  for (size_t p = 0; p < params.size(); p++)
    for (int i = 0; i < params[p]->getNValues(); i++) {
      before[p].push_back(params[p]->getValueDirect(i));
      grads[p].push_back(params[p]->getGradientDirect(i));
    }

  sgd.updateParameters();

  // Each value should move by -lr * gradient.
  for (size_t p = 0; p < params.size(); p++)
    for (int i = 0; i < params[p]->getNValues(); i++) {
      double expected = before[p][i] - lr * grads[p][i];
      if (std::abs(params[p]->getValueDirect(i) - expected) > TOL) return false;
    }
  return true;
}

bool testZeroGrad() {
  auto net = buildNet();
  auto params = net.getParameters();
  mattTorch::SGD sgd(params, 0.1);

  populateGradients(net);
  sgd.zeroGrad();

  for (auto& p : params)
    for (int i = 0; i < p->getNValues(); i++)
      if (p->getGradientDirect(i) != 0.0) return false;
  return true;
}

int main() {
  run("SGD step (theta - lr*grad)", testSGDStep());
  run("zeroGrad clears gradients", testZeroGrad());
}
