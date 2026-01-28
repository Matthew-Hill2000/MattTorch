


#include <mattTorch/criterion/criterion.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

namespace mattTorch::criterion {

class CrossEntropyLoss : public Criterion {
 public:
  TensorView calculateLoss(TensorView& input, TensorView& target);
};

}  // namespace mattTorch::criterion
