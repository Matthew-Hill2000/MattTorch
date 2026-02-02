#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

namespace mattTorch {

class GradFunction;

TensorView::TensorView()
    : gradient{nullptr},
      requiresGrad{true},
      hasGrad{false},
      dimensions{1},
      strides{1},
      offset{0},
      nValues{1},
      rank{1} {
  storage = std::make_shared<tensor::TensorStorage>(1);
  gradFunction =
      std::make_shared<function::GradAccumulator>(gradient, dimensions);
}

TensorView::TensorView(const std::vector<int>& dims, bool isLeaf)
    : gradient{nullptr},
      isLeaf{isLeaf},
      requiresGrad{true},
      hasGrad{true},
      dimensions{dims},
      strides{},
      offset{0},
      nValues{1},
      rank{static_cast<int>(dims.size())} {
  strides.resize(rank);

  if (rank == 0) {
    nValues = 0;
  } else {
    strides[rank - 1] = 1;
    nValues = dimensions[rank - 1];
    for (int i{rank - 2}; i >= 0; i--) {
      strides[i] = strides[i + 1] * dimensions[i + 1];
      nValues *= dimensions[i];
    }
  }

  storage = std::make_shared<tensor::TensorStorage>(nValues);
  if (isLeaf) {
    gradient = std::make_shared<TensorView>(dimensions, false);
    gradFunction =
        std::make_shared<function::GradAccumulator>(gradient, dimensions);
  }
}

TensorView::TensorView(const TensorView& other)
    : storage{other.storage},
      gradient{other.gradient},
      gradFunction{other.gradFunction},
      isLeaf{other.isLeaf},
      requiresGrad{other.requiresGrad},
      hasGrad{other.hasGrad},
      dimensions{other.dimensions},
      strides{other.strides},
      offset{other.offset},
      nValues{other.nValues},
      rank{other.rank} {}

// Creates a tensor with the specific arguments
TensorView::TensorView(std::shared_ptr<tensor::TensorStorage> storage,
                       std::vector<int> dims, std::vector<int> strides,
                       int offset, int nVals, int rnk,
                       std::shared_ptr<TensorView> gradient,
                       std::shared_ptr<GradFunction> gradFunction, bool isLeaf,
                       bool requiresGrad, bool hasGrad)
    : storage{storage},
      gradient{gradient},
      gradFunction{gradFunction},
      isLeaf{isLeaf},
      requiresGrad{requiresGrad},
      hasGrad{hasGrad},
      dimensions{dims},
      strides{strides},
      offset{offset},
      nValues{nVals},
      rank{rnk} {}

TensorView TensorView::deepCopy() const {
  TensorView result(this->dimensions);

  double* rhs = this->getData();
  double* lhs = result.getData();

  int i{0};
  for (; i + 3 < nValues; i += 4) {
    __m256d a = _mm256_loadu_pd(rhs + i);
    _mm256_storeu_pd(lhs + i, a);
  }

  for (; i < nValues; i++) {
    lhs[i] = rhs[i];
  }

  return result;
}

void TensorView::print(std::ostream& out) const {
  if (rank == 1) {
    // 1D tensor
    int width = 0;
    for (int i = 0; i < dimensions[0]; i++) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << this->getValueDirect(i);
      width = std::max(width, static_cast<int>(oss.str().length()));
    }

    out << "[";
    for (int i = 0; i < dimensions[0]; i++) {
      out << std::setw(width) << std::fixed << std::setprecision(2)
          << this->getValueDirect(i);
      if (i != dimensions[0] - 1) out << ", ";
    }
    out << "]\n";
  } else if (rank == 2) {
    this->printMatrix(out, 0);
  } else if (rank == 3) {
    this->printTripleMatrix(out);
  } else if (rank == 4) {
    this->printQuadrupleMatrix(out);
  }
}

void TensorView::printMatrix(std::ostream& out, int spaces) const {
  int width = 0;
  for (int i = 0; i < dimensions[0]; i++) {
    for (int j = 0; j < dimensions[1]; j++) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << this->getValue({i, j});
      width = std::max(width, static_cast<int>(oss.str().length()));
    }
  }

  for (int i = 0; i < dimensions[0]; i++) {
    for (int s = 0; s < spaces; s++) out << " ";
    out << "[";
    for (int j = 0; j < dimensions[1]; j++) {
      out << std::setw(width) << std::fixed << std::setprecision(2)
          << this->getValue({i, j});
      if (j != dimensions[1] - 1) out << ", ";
    }
    out << "]\n";
  }
}

void TensorView::printTripleMatrix(std::ostream& out) const {
  int width = 0;
  for (int i = 0; i < dimensions[0]; i++)
    for (int j = 0; j < dimensions[1]; j++)
      for (int k = 0; k < dimensions[2]; k++) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (*this)[{i, j, k}];
        width = std::max(width, static_cast<int>(oss.str().length()));
      }

  for (int i = 0; i < dimensions[1]; i++) {
    out << "[ ";
    for (int j = 0; j < dimensions[0]; j++) {
      out << "[";
      for (int k = 0; k < dimensions[2]; k++) {
        out << std::setw(width) << std::fixed << std::setprecision(2)
            << (*this)[{j, i, k}];
        if (k != dimensions[2] - 1) out << ", ";
      }
      out << "] ";
    }
    out << "]\n";
  }
}

void TensorView::printQuadrupleMatrix(std::ostream& out) const {
  int width = 0;
  for (int i = 0; i < dimensions[0]; i++)
    for (int j = 0; j < dimensions[1]; j++)
      for (int k = 0; k < dimensions[2]; k++)
        for (int l = 0; l < dimensions[3]; l++) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << (*this)[{i, j, k, l}];
          width = std::max(width, static_cast<int>(oss.str().length()));
        }

  for (int i = 0; i < dimensions[0]; i++) {
    for (int k = 0; k < dimensions[2]; k++) {
      out << "[ ";
      for (int j = 0; j < dimensions[1]; j++) {
        out << "[";
        for (int l = 0; l < dimensions[3]; l++) {
          out << std::setw(width) << std::fixed << std::setprecision(2)
              << (*this)[{i, j, k, l}];
          if (l != dimensions[3] - 1) out << ", ";
        }
        out << "] ";
      }
      out << "]\n";
    }
    out << "\n";
  }
}

std::ostream& operator<<(std::ostream& out, const TensorView& tensor) {
  tensor.print(out);
  return out;
}
}  // namespace mattTorch
