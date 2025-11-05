#include "function/gradFunction.h"
#include "tensor/tensor.h"
#include "tensorStorage/tensorStorage.h"
#include "tensorView/tensorView.h"

int main() {
  Tensor a({1});
  a[0] = 2.0;
  Tensor b({1});
  b[0] = 3.0;

  Tensor c = a * b;
  Tensor d = c.exponent(2);
}
