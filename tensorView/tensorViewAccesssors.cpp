#include "tensorView.h"

TensorView TensorView::operator[](int index) {
  if (index < 0 || index >= mDimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }
  if (mRank == 1) {
    return TensorView(mStorage, {1}, {1}, mOffset + index * mStrides[0], 1, 1);
  } else {
    std::vector<int> newDims(mDimensions.begin() + 1, mDimensions.end());
    std::vector<int> newStrides(mStrides.begin() + 1, mStrides.end());
    int newOffset(mOffset + index * mStrides[0]);
    int newNValues{mNValues / mDimensions[0]};
    int newRank{mRank - 1};

    return TensorView(mStorage, newDims, newStrides, newOffset, newNValues,
                      newRank);
  }
}

const TensorView TensorView::operator[](int index) const {
  if (index < 0 || index >= mDimensions[0]) {
    throw std::out_of_range("Index out of bounds for first dimension");
  }

  if (mRank == 1) {
    return TensorView(mStorage, {1}, {1}, mOffset + index * mStrides[0], 1, 1);
  } else {
    std::vector<int> newDims(mDimensions.begin() + 1, mDimensions.end());
    std::vector<int> newStrides(mStrides.begin() + 1, mStrides.end());
    int newOffset{mOffset + index * mStrides[0]};
    int newNValues{mNValues / mDimensions[0]};
    int newRank{mRank - 1};

    return TensorView(mStorage, newDims, newStrides, newOffset, newNValues,
                      newRank);
  }
}

double& TensorView::operator[](const std::vector<int>& rIndices) {
  int index = calculateIndex(rIndices);
  return mStorage->at(index);
}

const double& TensorView::operator[](const std::vector<int>& rIndices) const {
  int index = calculateIndex(rIndices);
  return mStorage->at(index);
}

int TensorView::calculateIndex(const std::vector<int>& rIndices) const {
  if (static_cast<int>(rIndices.size()) != mRank) {
    throw std::invalid_argument("Number of indices doesn't match tensor rank");
  }

  for (int i{0}; i < static_cast<int>(rIndices.size()); i++) {
    if (rIndices[i] < 0 || rIndices[i] >= mDimensions[i]) {
      throw std::out_of_range("Index out of bounds for dimension " +
                              std::to_string(i));
    }
  }

  int index{mOffset};
  for (int i{0}; i < mRank; i++) {
    index += rIndices[i] * mStrides[i];
  }

  return index;
}

void TensorView::setValue(const std::vector<int>& rIndices, double value) {
  int index = calculateIndex(rIndices);
  mStorage->at(index) = value;
}

double TensorView::getValue(const std::vector<int>& rIndices) const {
  int index = calculateIndex(rIndices);
  return mStorage->at(index);
}

void TensorView::setValueDirect(int linearIndex, double value) {
  if (linearIndex < 0 || linearIndex >= mNValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  mStorage->at(mOffset + linearIndex) = value;
}

double TensorView::getValueDirect(int linearIndex) const {
  if (linearIndex < 0 || linearIndex >= mNValues) {
    throw std::out_of_range("Linear index out of bounds");
  }
  return mStorage->at(mOffset + linearIndex);
}

std::vector<double> TensorView::getValues() const {
  std::vector<double> result(mNValues);

  for (int i{0}; i < mNValues; i++) {
    result[i] = getValueDirect(i);
  }

  return result;
}

std::vector<int> TensorView::getStrides() const { return mStrides; }

int TensorView::getOffset() const { return mOffset; }

int TensorView::getNValues() const { return mNValues; }

int TensorView::getRank() const { return mRank; }

std::vector<int> TensorView::getDimensions() const { return mDimensions; }
