
#pragma once

#include <mattTorch/network/network.h>
#include <mattTorch/tensor/tensorView/tensorView.h>
#include <mattTorch/optimiser/optimiser.h>

#include <memory>
#include <vector>

namespace mattTorch {

class SGD : public Optimiser {
 private:
  std::vector<std::shared_ptr<TensorView>> parameters;
  double learningRate;

 public:
  SGD(std::vector<std::shared_ptr<TensorView>> parameters, double learningRate);
  void updateParameters();
  void zeroGrad();
};
}  // namespace mattTorch::optimiser
