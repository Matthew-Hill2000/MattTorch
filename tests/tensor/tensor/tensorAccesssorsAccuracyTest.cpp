#include <mattTorch/mattTorch.h>

#include "common/testUtils.h"

bool testValueRoundtrip() {
  mattTorch::Tensor t({3, 4});
  t.setRequiresGrad(false);
  t.setValue({1, 2}, 3.14);
  if (t.getValue({1, 2}) != 3.14) return false;
  // Row-major linear index of {1,2} in a {3,4} tensor is 1*4 + 2 = 6.
  if (t.getValueDirect(6) != 3.14) return false;
  t.setValueDirect(6, 2.71);
  return t.getValue({1, 2}) == 2.71;
}

bool testStridesRowMajor() {
  mattTorch::Tensor t({3, 4});
  auto s = t.getStrides();
  return s[0] == 4 && s[1] == 1;
}

bool testCalculateIndexViaSubscript() {
  mattTorch::Tensor t({2, 3, 4});
  t.setRequiresGrad(false);
  // &t[{i,j,k}] writes to storage at i*12 + j*4 + k.
  t[{1, 2, 3}] = 9.0;
  return t.getValueDirect(1 * 12 + 2 * 4 + 3) == 9.0;
}

bool testRowIndexing() {
  mattTorch::Tensor a = randomTensor({3, 4}, false);
  mattTorch::Tensor row = a[1];
  if (static_cast<int>(row.getRank()) != 1) return false;
  for (int j = 0; j < 4; j++)
    if (row.getValueDirect(j) != a.getValue({1, j})) return false;
  return true;
}

bool testMetadata() {
  mattTorch::Tensor t({2, 3, 4});
  return t.getNValues() == 24 && static_cast<int>(t.getRank()) == 3 &&
         t.getOffset() == 0;
}

int main() {
  run("Value roundtrip", testValueRoundtrip());
  run("Row-major strides", testStridesRowMajor());
  run("calculateIndex via []", testCalculateIndexViaSubscript());
  run("Row indexing", testRowIndexing());
  run("Metadata", testMetadata());
}
