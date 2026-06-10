#include <mattTorch/mattTorch.h>
#include <mattTorch/network/network.h>

#include <cmath>
#include <cstdio>

#include "common/testUtils.h"

mattTorch::Network buildNet() {
  return mattTorch::NetworkBuilder()
      .addFullyConnectedLayer(8, 16)
      .addTanhLayer()
      .addFullyConnectedLayer(16, 4)
      .build();
}

bool testForwardShape() {
  auto net = buildNet();
  mattTorch::Tensor input = randomTensor({5, 8}, false);
  mattTorch::Tensor out = net.forward(input);
  auto d = out.getDimensions();
  return d[0] == 5 && d[1] == 4;
}

bool testParameterCount() {
  auto net = buildNet();
  return net.getParameters().size() == 4;
}

bool testSaveLoadRoundTrip() {
  auto net1 = buildNet();
  const std::string path = "network_params_test.bin";
  net1.saveParameters(path);

  auto net2 = buildNet();
  net2.loadParameters(path);
  std::remove(path.c_str());

  auto p1 = net1.getParameters();
  auto p2 = net2.getParameters();
  if (p1.size() != p2.size()) return false;
  for (size_t k = 0; k < p1.size(); k++) {
    if (p1[k]->getNValues() != p2[k]->getNValues()) return false;
    for (int i = 0; i < p1[k]->getNValues(); i++) {
      const double a = p1[k]->getValueDirect(i);
      const double b = p2[k]->getValueDirect(i);
      if (std::abs(a - b) > 1e-4 * (1.0 + std::abs(a))) return false;
    }
  }
  return true;
}

int main() {
  run("Forward shape", testForwardShape());
  run("Parameter count", testParameterCount());
  run("Save/load round-trip", testSaveLoadRoundTrip());
}
