#include <mattTorch/mattTorch.h>
#include <mattTorch/network/network.h>

#include "common/testUtils.h"

// ============ Tests ============

void benchNetworkForward(mattTorch::Network& net, mattTorch::Tensor& input) {
  mattTorch::Tensor out = net.forward(input);
  out.backward();
}

int main() {
  const int batch = 1024;
  auto net = mattTorch::NetworkBuilder()
                 .addFullyConnectedLayer(1024, 1024)
                 .addReluLayer()
                 .addFullyConnectedLayer(1024, 1024)
                 .addTanhLayer()
                 .addFullyConnectedLayer(1024, 1024)
                 .build();
  mattTorch::Tensor input = randomTensor({batch, 1024}, false);

  run("Network forward (3 FC)", [&] { benchNetworkForward(net, input); });
}
