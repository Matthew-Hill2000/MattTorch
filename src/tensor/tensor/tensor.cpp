#include <mattTorch/function/accumulator/gradAccumulator.h>
#include <mattTorch/tensor/tensor/tensor.h>
#include <mattTorch/tensor/tensorStorage/tensorStorage.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <memory>
#include <plotlypp/traits.hpp>
#include <sstream>

#include "mattTorch/tensor/tensor/tensorTypes.h"

namespace mattTorch {

Tensor::Tensor()
    : gradient{nullptr},
      tensorData{Dims{1}, Strides{1}, 0},
      gradData{.isLeaf = true, .requiresGrad = true, .hasGrad = false} {
  storage = std::make_shared<tensor::TensorStorage>(1);

  gradFunction = std::make_shared<function::GradAccumulator>(
      gradient, tensorData.dimensions);
}

Tensor::Tensor(const Dims& dims, bool isLeaf)
    : gradient{nullptr},
      tensorData{dims, Strides(), 0},
      gradData{.isLeaf = isLeaf, .requiresGrad = isLeaf, .hasGrad = false} {
  tensorData.strides.resize(tensorData.dimensions.size());

  const std::size_t rank = tensorData.dimensions.size();

  if (rank != 0) {
    tensorData.strides[rank - 1] = 1;
    for (std::size_t i = rank - 1; i > 0; i--) {
      tensorData.strides[i - 1] =
          tensorData.strides[i] * tensorData.dimensions[i];
    }
  }

  int nValues = tensorData.dimensions.product();

  storage = std::make_shared<tensor::TensorStorage>(nValues);

  if (isLeaf) {
    gradient = std::make_shared<Tensor>(tensorData.dimensions, false);
    gradFunction = std::make_shared<function::GradAccumulator>(
        gradient, tensorData.dimensions);
  } else {
    gradFunction = nullptr;
  }
}

Tensor::Tensor(std::shared_ptr<tensor::TensorStorage> storage,
               TensorData tensorData, std::shared_ptr<Tensor> gradient,
               std::shared_ptr<GradFunction> gradFunction, GradData gradData)
    : storage{std::move(storage)},
      gradient{std::move(gradient)},
      gradFunction{std::move(gradFunction)},
      tensorData{std::move(tensorData)},
      gradData{std::move(gradData)} {}

Tensor Tensor::deepCopy() const {
  Tensor result(this->tensorData.dimensions);
  const int nValues = getNValues();

  double* rhs = this->getData();
  double* lhs = result.getData();

  std::memcpy(lhs, rhs, nValues * sizeof(double));

  return result;
}

void Tensor::print(std::ostream& out) const {
  const std::size_t rank = getRank();
  if (rank == 1) {
    // 1D tensor
    int width = 0;
    for (int i = 0; i < tensorData.dimensions[0]; i++) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << this->getValueDirect(i);
      width = std::max(width, static_cast<int>(oss.str().length()));
    }

    out << "[";
    for (int i = 0; i < tensorData.dimensions[0]; i++) {
      out << std::setw(width) << std::fixed << std::setprecision(2)
          << this->getValueDirect(i);
      if (i != tensorData.dimensions[0] - 1) {
        out << ", ";
      }
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

void Tensor::printMatrix(std::ostream& out, int spaces) const {
  int width = 0;
  for (int i = 0; i < tensorData.dimensions[0]; i++) {
    for (int j = 0; j < tensorData.dimensions[1]; j++) {
      std::ostringstream oss;
      oss << std::fixed << std::setprecision(2) << this->getValue({i, j});
      width = std::max(width, static_cast<int>(oss.str().length()));
    }
  }

  for (int i = 0; i < tensorData.dimensions[0]; i++) {
    for (int s = 0; s < spaces; s++) {
      out << " ";
    }
    out << "[";
    for (int j = 0; j < tensorData.dimensions[1]; j++) {
      out << std::setw(width) << std::fixed << std::setprecision(2)
          << this->getValue({i, j});
      if (j != tensorData.dimensions[1] - 1) {
        out << ", ";
      }
    }
    out << "]\n";
  }
}

void Tensor::printTripleMatrix(std::ostream& out) const {
  int width = 0;
  for (int i = 0; i < tensorData.dimensions[0]; i++) {
    for (int j = 0; j < tensorData.dimensions[1]; j++) {
      for (int k = 0; k < tensorData.dimensions[2]; k++) {
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << (*this)[{i, j, k}];
        width = std::max(width, static_cast<int>(oss.str().length()));
      }
    }
  }

  for (int i = 0; i < tensorData.dimensions[1]; i++) {
    out << "[ ";
    for (int j = 0; j < tensorData.dimensions[0]; j++) {
      out << "[";
      for (int k = 0; k < tensorData.dimensions[2]; k++) {
        out << std::setw(width) << std::fixed << std::setprecision(2)
            << (*this)[{j, i, k}];
        if (k != tensorData.dimensions[2] - 1) {
          out << ", ";
        }
      }
      out << "] ";
    }
    out << "]\n";
  }
}

void Tensor::printQuadrupleMatrix(std::ostream& out) const {
  int width = 0;
  for (int i = 0; i < tensorData.dimensions[0]; i++) {
    for (int j = 0; j < tensorData.dimensions[1]; j++) {
      for (int k = 0; k < tensorData.dimensions[2]; k++) {
        for (int l = 0; l < tensorData.dimensions[3]; l++) {
          std::ostringstream oss;
          oss << std::fixed << std::setprecision(2) << (*this)[{i, j, k, l}];
          width = std::max(width, static_cast<int>(oss.str().length()));
        }
      }
    }
  }

  for (int i = 0; i < tensorData.dimensions[0]; i++) {
    for (int k = 0; k < tensorData.dimensions[2]; k++) {
      out << "[ ";
      for (int j = 0; j < tensorData.dimensions[1]; j++) {
        out << "[";
        for (int l = 0; l < tensorData.dimensions[3]; l++) {
          out << std::setw(width) << std::fixed << std::setprecision(2)
              << (*this)[{i, j, k, l}];
          if (l != tensorData.dimensions[3] - 1) {
            out << ", ";
          }
        }
        out << "] ";
      }
      out << "]\n";
    }
    out << "\n";
  }
}

std::ostream& operator<<(std::ostream& out, const Tensor& tensor) {
  tensor.print(out);
  return out;
}
}  // namespace mattTorch
   //
