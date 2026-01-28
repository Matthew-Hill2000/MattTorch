#pragma once

#include <mattTorch/network/network.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <memory>
#include <vector>

namespace mattTorch {

class Optimiser {
 private:
  std::vector<std::shared_ptr<TensorView>> parameters;
  double learningRate;
  
public:
  void updateParameters(); 
  void zeroGrad();
  virtual ~Optimiser() = default;

};
}  // namespace mattTorch
