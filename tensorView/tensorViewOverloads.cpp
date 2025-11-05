#include "tensorView.h"

TensorView& TensorView::operator=(const TensorView& rOther) {
  if (this != &rOther) {
    mStorage = rOther.mStorage;
    mDimensions = rOther.mDimensions;
    mStrides = rOther.mStrides;
    mOffset = rOther.mOffset;
    mNValues = rOther.mNValues;
    mRank = rOther.mRank;
  }
  return *this;
}

TensorView& TensorView::operator=(double val) {
  for (int i = 0; i < mNValues; i++) {
    setValueDirect(i, val);
  }
  return *this;
}

TensorView TensorView::operator+(const TensorView& rOther) const {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument("Tensor dimensions must match for addition");
  }

  TensorView result(mDimensions);

  for (int i = 0; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) + rOther.getValueDirect(i));
  }

  return result;
}

TensorView TensorView::operator-(const TensorView& rOther) const {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument("Tensor dimensions must match for subtraction");
  }

  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) - rOther.getValueDirect(i));
  }

  return result;
}

TensorView TensorView::operator*(const TensorView& rOther) const {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise multiplication");
  }

  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) * rOther.getValueDirect(i));
  }

  return result;
}

TensorView TensorView::operator/(const TensorView& rOther) const {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument("Tensor dimensions must match for division");
  }

  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    double divisor{rOther.getValueDirect(i)};
    if (divisor == 0.0) {
      throw std::invalid_argument("Division by zero");
    }
    result.setValueDirect(i, getValueDirect(i) / divisor);
  }

  return result;
}

TensorView& TensorView::operator+=(const TensorView& rOther) {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for += operation");
  }

  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) + rOther.getValueDirect(i));
  }
  return *this;
}

TensorView& TensorView::operator-=(const TensorView& rOther) {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for -= operation");
  }

  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) - rOther.getValueDirect(i));
  }

  return *this;
}

TensorView& TensorView::operator*=(const TensorView& rOther) {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for *= operation");
  }

  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) * rOther.getValueDirect(i));
  }

  return *this;
}

TensorView& TensorView::operator/=(const TensorView& rOther) {
  if (mDimensions != rOther.mDimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for /= operation");
  }

  for (int i{0}; i < mNValues; i++) {
    double divisor{rOther.getValueDirect(i)};
    if (divisor == 0.0) {
      throw std::invalid_argument("Division by zero");
    }
    setValueDirect(i, getValueDirect(i) / divisor);
  }

  return *this;
}

TensorView TensorView::operator+(double scalar) const {
  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) + scalar);
  }

  return result;
}

TensorView TensorView::operator-(double scalar) const {
  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) - scalar);
  }

  return result;
}

TensorView TensorView::operator*(double scalar) const {
  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) * scalar);
  }

  return result;
}

TensorView TensorView::operator/(double scalar) const {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }

  TensorView result(mDimensions);

  for (int i{0}; i < mNValues; i++) {
    result.setValueDirect(i, getValueDirect(i) / scalar);
  }

  return result;
}

TensorView& TensorView::operator+=(double scalar) {
  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) + scalar);
  }

  return *this;
}

TensorView& TensorView::operator-=(double scalar) {
  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) - scalar);
  }

  return *this;
}

TensorView& TensorView::operator*=(double scalar) {
  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) * scalar);
  }

  return *this;
}

TensorView& TensorView::operator/=(double scalar) {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }

  for (int i{0}; i < mNValues; i++) {
    setValueDirect(i, getValueDirect(i) / scalar);
  }

  return *this;
}

bool TensorView::operator==(const TensorView& rOther) const {
  if (mDimensions != rOther.mDimensions) {
    return false;
  }

  const double epsilon{1e-9};
  for (int i{0}; i < mNValues; i++) {
    if (std::abs(getValueDirect(i) - rOther.getValueDirect(i)) > epsilon) {
      return false;
    }
  }

  return true;
}

bool TensorView::operator!=(const TensorView& rOther) const {
  return !(*this == rOther);
}
