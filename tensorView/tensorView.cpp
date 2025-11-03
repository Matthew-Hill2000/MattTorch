#include "tensorView.h"

#include <string>
#include <vector>

TensorView::TensorView()
    : mDimensions{1}, mStrides{1}, mOffset(0), mNValues(1), mRank(1) {
  mStorage = std::make_shared<TensorStorage>(1, 0.0);
}

TensorView::TensorView(const std::vector<int>& dims)
    : mDimensions(dims),
      mStrides(),
      mOffset(0),
      mNValues(1),
      mRank(dims.size()) {
  mStrides.resize(mRank);
  if (mRank == 0) {
    mNValues = 0;
  } else if (mRank == 1) {
    mNValues = mDimensions[0];
    mStrides[0] = 1;
  } else {
    mStrides[mRank - 1] = 1;
    mNValues = mDimensions[mRank - 1];
    for (int i(mRank - 2); i >= 0; i--) {
      mStrides[i] = mStrides[i + 1] * mDimensions[i + 1];
      mNValues *= mDimensions[i];
    }
  }
  mStorage = std::make_shared<TensorStorage>(mNValues, 0.0);
}

TensorView::TensorView(const TensorView& rOther)
    : mStorage(rOther.mStorage),
      mDimensions(rOther.mDimensions),
      mStrides(rOther.mStrides),
      mOffset(rOther.mOffset),
      mNValues(rOther.mNValues),
      mRank(rOther.mRank) {}

TensorView::TensorView(std::shared_ptr<TensorStorage> storage,
                       std::vector<int> dims, std::vector<int> strides,
                       int offset, int nVals, int rnk)
    : mStorage(storage),
      mDimensions(dims),
      mStrides(strides),
      mOffset(offset),
      mNValues(nVals),
      mRank(rnk) {}
TensorView TensorView::deepCopy() const {
  TensorView result(mDimensions);
  for (int i = 0; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i));
  }
  return result;
}

void TensorView::print(int index, int dim, int indent) const {
  std::string indentation(indent, ' ');
  std::cout << indentation << "[";
  if (dim == mRank - 1) {
    for (int i{0}; i < mDimensions[dim]; i++) {
      std::vector<int> indices(mRank);
      for (int j{0}; j < dim; j++) {
        indices[j] = (index / mStrides[j]) % mDimensions[j];
      }
      indices[dim] = i;
      std::cout << getValue(indices);
      if (i < mDimensions[dim] - 1) {
        std::cout << ", ";
      }
    }
  } else {
    std::cout << std::endl;
    for (int i{0}; i < mDimensions[dim]; i++) {
      int nextIndex{index + i * mStrides[dim]};
      print(nextIndex, dim + 1, indent + 2);
      if (i < mDimensions[dim] - 1) {
        std::cout << "," << std::endl;
      }
    }
    std::cout << std::endl << indentation;
  }
  std::cout << "]";
  if (dim == 0) {
    std::cout << std::endl;
  }
}
