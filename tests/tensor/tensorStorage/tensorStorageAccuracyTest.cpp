#include <mattTorch/mattTorch.h>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>

#include "common/testUtils.h"

using mattTorch::tensor::TensorStorage;

bool testSizeAndZeroInit() {
  TensorStorage s(10);
  if (s.getSize() != 10) return false;
  for (int i = 0; i < 10; i++)
    if (s.at(i) != 0.0) return false;
  return true;
}

bool testReadWriteAndGetData() {
  TensorStorage s(8);
  for (int i = 0; i < 8; i++) s.at(i) = i * 1.5;
  for (int i = 0; i < 8; i++)
    if (s.at(i) != i * 1.5) return false;
  double* d = s.getData();
  d[3] = 99.0;
  return s.at(3) == 99.0;
}

bool testSetAllValues() {
  TensorStorage s(16);
  s.setAllValues(7.0);
  for (int i = 0; i < 16; i++)
    if (s.at(i) != 7.0) return false;
  return true;
}

int main() {
  run("Size & zero-init", testSizeAndZeroInit());
  run("Read/write & getData", testReadWriteAndGetData());
  run("setAllValues", testSetAllValues());
}
