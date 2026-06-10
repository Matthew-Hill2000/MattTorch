#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

bool testConstruction() {
  mattTorch::Tensor t({3, 4});
  if (t.getNValues() != 12) return false;
  if (static_cast<int>(t.getRank()) != 2) return false;
  auto d = t.getDimensions();
  return d[0] == 3 && d[1] == 4;
}

bool testFreshTensorIsZeroed() {
  mattTorch::Tensor t({5, 5});
  for (int i = 0; i < t.getNValues(); i++)
    if (t.getValueDirect(i) != 0.0) return false;
  return true;
}

bool testDeepCopyIsIndependent() {
  mattTorch::Tensor a = randomTensor({4, 4}, false);
  mattTorch::Tensor b = a.deepCopy();
  if (!(a == b)) return false;
  b.setValueDirect(0, a.getValueDirect(0) + 1.0);
  return a.getValueDirect(0) != b.getValueDirect(0);
}

bool testRequiresGradFlag() {
  mattTorch::Tensor t({2, 2});
  t.setRequiresGrad(true);
  if (!t.getRequiresGrad()) return false;
  t.setRequiresGrad(false);
  return !t.getRequiresGrad();
}

int main() {
  run("Construction", testConstruction());
  run("Fresh tensor zeroed", testFreshTensorIsZeroed());
  run("DeepCopy independent", testDeepCopyIsIndependent());
  run("RequiresGrad flag", testRequiresGradFlag());
}
