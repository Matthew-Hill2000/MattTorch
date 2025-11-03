#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <memory>
#include <vector>

class TensorStorage;

class TensorView {
 private:
  std::shared_ptr<TensorStorage> mStorage;

  std::vector<int> mDimensions;
  std::vector<int> mStrides;
  int mOffset;
  int mNValues;
  int mRank;

  int calculateIndex(const std::vector<int>& rIndices) const;

 public:
  TensorView();
  TensorView(const std::vector<int>& dims);
  TensorView(const TensorView& rOther);
  TensorView(std::shared_ptr<TensorStorage> storage, std::vector<int> dims,
             std::vector<int> strides, int offset, int nVals, int rnk);

  TensorView& operator=(const TensorView& rOther);
  TensorView& operator=(double val);

  void setValue(const std::vector<int>& rIndices, double value);
  double getValue(const std::vector<int>& rIndices) const;
  void setValueDirect(int index, double value);
  double getValueDirect(int index) const;

  TensorView operator[](int index);
  const TensorView operator[](int index) const;
  double& operator[](const std::vector<int>& rIndices);
  const double& operator[](const std::vector<int>& rIndices) const;

  TensorView matrixMultiply(const TensorView& rRhs) const;
  TensorView elementwiseProduct(const TensorView& rRhs) const;
  TensorView transpose() const;
  TensorView crossCorrelate(const TensorView& rKernel) const;
  TensorView fullyConvolve(const TensorView& rKernel) const;
  TensorView convolve(const TensorView& rKernel) const;

  TensorView operator+(const TensorView& rOther) const;
  TensorView operator-(const TensorView& rOther) const;
  TensorView operator*(const TensorView& rOther) const;
  TensorView operator/(const TensorView& rOther) const;
  TensorView& operator+=(const TensorView& rOther);
  TensorView& operator-=(const TensorView& rOther);
  TensorView& operator*=(const TensorView& rOther);
  TensorView& operator/=(const TensorView& rOther);

  TensorView operator+(double scalar) const;
  TensorView operator-(double scalar) const;
  TensorView operator*(double scalar) const;
  TensorView operator/(double scalar) const;
  TensorView& operator+=(double scalar);
  TensorView& operator-=(double scalar);
  TensorView& operator*=(double scalar);
  TensorView& operator/=(double scalar);

  bool operator==(const TensorView& rOther) const;
  bool operator!=(const TensorView& rOther) const;

  TensorView deepCopy() const;

  void print(int index = 0, int dim = 0, int indent = 0) const;

  std::vector<double> getValues() const;
  std::vector<int> getStrides() const;
  int getOffset() const;
  int getNValues() const;
  int getRank() const;
  std::vector<int> getDimensions() const;
};

#endif
