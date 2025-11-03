#include <omp.h>

#include "tensorView.h"

TensorView TensorView::matrixMultiply(const TensorView& rRhs) const {
  if (mRank != 2 || rRhs.mRank != 2 || mDimensions[1] != rRhs.mDimensions[0]) {
    throw std::invalid_argument("Invalid dimensions for matrix multiplication");
  }

  std::vector<int> resultDims = {mDimensions[0], rRhs.mDimensions[1]};
  TensorView result(resultDims);

  for (int i = 0; i < mDimensions[0]; i++) {
    for (int j = 0; j < rRhs.mDimensions[1]; j++) {
      double sum{0.0};
      for (int k = 0; k < mDimensions[1]; k++) {
        sum += getValue({i, k}) * rRhs.getValue({k, j});
      }
      result.setValue({i, j}, sum);
    }
  }

  return result;
}

TensorView TensorView::elementwiseProduct(const TensorView& rRhs) const {
  if (mDimensions != rRhs.mDimensions) {
    throw std::invalid_argument(
        "Dimensions must match for elementwise multiplication");
  }

  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) * rRhs.getValueDirect(i));
  }

  return result;
}

TensorView TensorView::transpose() const {
  if (mRank <= 1) {
    return *this;
  }

  std::vector<int> transposedDims(mDimensions.rbegin(), mDimensions.rend());
  std::vector<int> transposedStrides(mStrides.rbegin(), mStrides.rend());

  return TensorView(mStorage, transposedDims, transposedStrides, mOffset,
                    mNValues, mRank);
}

TensorView TensorView::crossCorrelate(const TensorView& rKernel) const {
  if (mRank != rKernel.mRank) {
    throw std::runtime_error("Tensor and kernel must have same rank");
  }

  if (mRank != 2) {
    throw std::runtime_error(
        "Cross-correlation is only implemented for 2D tensors");
  }

  std::vector<int> outputDims;
  for (int i{0}; i < mRank; i++) {
    outputDims.push_back(mDimensions[i] - rKernel.mDimensions[i] + 1);
    if (outputDims[i] <= 0) {
      throw std::runtime_error("Kernel too large for dimension " +
                               std::to_string(i));
    }
  }

  TensorView result(outputDims);

  for (int i = 0; i < outputDims[0]; i++) {
    for (int j = 0; j < outputDims[1]; j++) {
      double sum{0.0};
      for (int ki = 0; ki < rKernel.mDimensions[0]; ki++) {
        for (int kj = 0; kj < rKernel.mDimensions[1]; kj++) {
          sum += getValue({i + ki, j + kj}) * rKernel.getValue({ki, kj});
        }
      }
      result.setValue({i, j}, sum);
    }
  }

  return result;
}

TensorView TensorView::fullyConvolve(const TensorView& rKernel) const {
  if (mRank != rKernel.mRank) {
    throw std::runtime_error("Tensor and kernel must have same rank");
  }

  if (mRank != 2) {
    throw std::runtime_error(
        "Full convolution is only implemented for 2D tensors");
  }

  std::vector<int> outputDims;
  for (int i{0}; i < mRank; i++) {
    outputDims.push_back(mDimensions[i] + rKernel.mDimensions[i] - 1);
  }

  TensorView result(outputDims);

  for (int i = 0; i < outputDims[0]; i++) {
    for (int j = 0; j < outputDims[1]; j++) {
      double sum{0.0};
      for (int ki = 0; ki < rKernel.mDimensions[0]; ki++) {
        for (int kj = 0; kj < rKernel.mDimensions[1]; kj++) {
          int x{i - ki};
          int y{j - kj};
          if (x >= 0 && x < mDimensions[0] && y >= 0 && y < mDimensions[1]) {
            sum += getValue({x, y}) * rKernel.getValue({ki, kj});
          }
        }
      }
      result.setValue({i, j}, sum);
    }
  }
  return result;
}

TensorView TensorView::convolve(const TensorView& rKernel) const {
  if (mRank != rKernel.mRank) {
    throw std::runtime_error("Tensor and kernel must have same rank");
  }

  if (mRank != 2) {
    throw std::runtime_error("Convolution is only implemented for 2D tensors");
  }

  std::vector<int> outputDims;
  for (int i{0}; i < mRank; i++) {
    outputDims.push_back(mDimensions[i] - rKernel.mDimensions[i] + 1);
    if (outputDims[i] <= 0) {
      throw std::runtime_error("Kernel too large for dimension " +
                               std::to_string(i));
    }
  }

  TensorView result(outputDims);

  for (int i = 0; i < outputDims[0]; i++) {
    for (int j = 0; j < outputDims[1]; j++) {
      double sum{0.0};
      for (int ki = 0; ki < rKernel.mDimensions[0]; ki++) {
        for (int kj = 0; kj < rKernel.mDimensions[1]; kj++) {
          sum += getValue({i + ki, j + kj}) *
                 rKernel.getValue({rKernel.mDimensions[0] - 1 - ki,
                                   rKernel.mDimensions[1] - 1 - kj});
        }
      }
      result.setValue({i, j}, sum);
    }
  }

  return result;
}
