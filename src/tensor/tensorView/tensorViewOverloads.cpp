#include <mattTorch/function/functionInclude.h>
#include <mattTorch/tensor/kernels/kernels.h>
#include <mattTorch/tensor/tensorView/tensorView.h>

#include <cmath>
#include <memory>
#include <stdexcept>

#include "mattTorch/function/exponential/gradExponential.h"
#include "mattTorch/function/log/gradLog.h"
#include "mattTorch/function/mean/gradMean.h"
#include "mattTorch/function/tanh/gradTanh.h"

namespace mattTorch {
TensorView& TensorView::operator=(const TensorView& other) {
  if (this != &other) {
    storage = other.storage;
    dimensions = other.dimensions;
    strides = other.strides;
    offset = other.offset;
    nValues = other.nValues;
    rank = other.rank;
    gradFunction = other.gradFunction;
    isLeaf = other.isLeaf;
    hasGrad = other.hasGrad;
    requiresGrad = other.requiresGrad;
    gradient = other.gradient;
  }
  return *this;
}

TensorView& TensorView::operator=(double val) {
  this->storage->setAllValues(val);
  return *this;
}

TensorView TensorView::operator+(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument("Tensor dimensions must match for addition");
  }

  TensorView result(dimensions, isLeaf = false);

  tensor::kernels::cpu::elementwiseAdd(this->getData(), other.getData(),
                                       result.getData(), this->nValues);

  if (!this->requiresGrad && !other.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradAdd>(savedTensors, nextFunctions));
  }

  return result;
}

TensorView TensorView::operator-(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument("Tensor dimensions must match for subtraction");
  }

  TensorView result(dimensions, false);

  tensor::kernels::cpu::elementwiseSubtract(this->getData(), other.getData(),
                                            result.getData(), this->nValues);

  if (!this->requiresGrad && !other.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradSubtract>(savedTensors, nextFunctions));
  }

  return result;
}

TensorView TensorView::operator*(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise multiplication");
  }

  TensorView result(dimensions, false);

  tensor::kernels::cpu::elementwiseMultiplication(
      this->getData(), other.getData(), result.getData(), this->nValues);

  if (!this->requiresGrad && !other.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this, other};

    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradMultiply>(savedTensors, nextFunctions));
  }
  return result;
}

TensorView TensorView::operator/(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument("Tensor dimensions must match for division");
  }

  TensorView result(dimensions, false);

  double* divisor = other.getData();
  for (int i{0}; i < nValues; i++) {
    if (divisor[i] == 0.0) {
      throw std::invalid_argument("Division by zero");
    }
  }

  tensor::kernels::cpu::elementwiseDivision(this->getData(), other.getData(),
                                            result.getData(), this->nValues);

  if (!this->requiresGrad && !other.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradDivide>(savedTensors, nextFunctions));
  }

  return result;
}

TensorView TensorView::operator+(double scalar) {
  TensorView result(dimensions, false);

  tensor::kernels::cpu::tensorScalarAdd(this->getData(), scalar,
                                        result.getData(), this->nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradAddScalar>(scalar, nextFunctions));
  }

  return result;
}

TensorView TensorView::operator-(double scalar) {
  TensorView result(dimensions, false);

  tensor::kernels::cpu::tensorScalarSubtract(this->getData(), scalar,
                                             result.getData(), this->nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradSubtractScalar>(scalar, nextFunctions));
  }
  return result;
}

TensorView TensorView::operator*(double scalar) {
  TensorView result(dimensions, false);

  tensor::kernels::cpu::tensorScalarMultiplication(
      this->getData(), scalar, result.getData(), this->nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradMultiplyScalar>(scalar, nextFunctions));
  }
  return result;
}

TensorView TensorView::operator/(double scalar) {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }

  TensorView result(dimensions, false);

  tensor::kernels::cpu::tensorScalarDivision(this->getData(), scalar,
                                             result.getData(), this->nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(std::make_shared<function::GradDivideScalar>(
        scalar, std::make_shared<TensorView>(*this), nextFunctions, true));
  }
  return result;
}

TensorView TensorView::elementwiseExponent(int scalar) {
  if (scalar == 0.0) {
    return *this;
  }

  TensorView result(dimensions, false);

  tensor::kernels::cpu::elementwiseExponent(this->getData(), scalar,
                                            result.getData(), nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<int> savedScalars{scalar};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    std::vector<TensorView> savedTensors{*this};

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(std::make_shared<function::GradExponent>(
        savedScalars, savedTensors, nextFunctions));
  }
  return result;
}

TensorView TensorView::reductionSum(int dim) {
  if (dim < 0 || dim >= rank) {
    throw std::invalid_argument("Invalid dimension for reduction");
  }

  // Build new dimensions with the reduced dim removed
  std::vector<int> newDimensions;
  for (int i = 0; i < rank; i++) {
    if (i != dim) {
      newDimensions.push_back(dimensions[i]);
    }
  }
  if (newDimensions.empty()) {
    newDimensions.push_back(1);
  }

  TensorView result(newDimensions, false);

  // Calculate strides for the kernel
  int outerSize = 1;
  for (int i = 0; i < dim; i++) {
    outerSize *= dimensions[i];
  }

  int reduceSize = dimensions[dim];

  int innerSize = 1;
  for (int i = dim + 1; i < rank; i++) {
    innerSize *= dimensions[i];
  }

  tensor::kernels::cpu::reductionSum(this->getData(), result.getData(),
                                     outerSize, reduceSize, innerSize);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    result.setGradFunction(std::make_shared<function::GradReductionSum>(
        savedTensors, dim, nextFunctions));
  }
  return result;
}

TensorView TensorView::mean() {
  TensorView result({1});

  tensor::kernels::cpu::mean(this->getData(), result.getData(), nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradMean>(*this, nextFunctions));
  }
  return result;
}

TensorView TensorView::log() {
  TensorView result(dimensions);

  tensor::kernels::cpu::log(this->getData(), result.getData(), nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradLog>(*this, nextFunctions));
  }
  return result;
}

TensorView TensorView::matrixMultiply(const TensorView& other) {
  if (dimensions.size() != 2 || other.dimensions.size() != 2) {
    throw std::invalid_argument("Tensors should be of rank 2");
  }

  if (dimensions[1] != other.dimensions[0]) {
    throw std::invalid_argument("Tensors innermost dimensions should match");
  }

  TensorView result(std::vector{dimensions[0], other.dimensions[1]}, false);

  tensor::kernels::cpu::matrixMultiplicationBlockVector(
      this->getData(), other.getData(), result.getData(), this->dimensions[0],
      this->dimensions[1], other.dimensions[1]);

  if (!this->requiresGrad && !other.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(std::make_shared<function::GradMultiplyMatrix>(
        savedTensors, nextFunctions));
  }
  return result;
}

TensorView TensorView::transposeMultiply(const TensorView& other,
                                         bool transposeFirst) {
  if (dimensions.size() != 2 || other.dimensions.size() != 2) {
    throw std::invalid_argument("Tensors should be of rank 2");
  }

  if (transposeFirst && (dimensions[0] != other.dimensions[0])) {
    throw std::invalid_argument("Tensors innermost dimensions should match");
  } else if (!transposeFirst && dimensions[1] != other.dimensions[1]) {
    throw std::invalid_argument("Tensors innermost dimensions should match");
  }

  TensorView result;
  if (transposeFirst) {
    result = TensorView(std::vector{dimensions[1], other.dimensions[1]}, false);

    tensor::kernels::cpu::transposeMultiplicationBlockVectorLHS(
        this->getData(), other.getData(), result.getData(), this->dimensions[0],
        this->dimensions[1], other.dimensions[1]);
  } else {
    result = TensorView(std::vector{dimensions[0], other.dimensions[0]}, false);

    tensor::kernels::cpu::transposeMultiplicationBlockVectorRHS(
        this->getData(), other.getData(), result.getData(), this->dimensions[0],
        this->dimensions[1], other.dimensions[0]);
  }

  if (!this->requiresGrad && !other.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<TensorView> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    if (transposeFirst) {
      result.setGradFunction(std::make_shared<function::GradTransposeMatrix>(
          savedTensors, nextFunctions, true));
    } else {
      result.setGradFunction(std::make_shared<function::GradTransposeMatrix>(
          savedTensors, nextFunctions, false));
    }
  }
  return result;
}

TensorView TensorView::ReLU() {
  TensorView result(dimensions);
  TensorView backwardsMask(dimensions);
  backwardsMask.setRequiresGrad(false);
  backwardsMask = 1.0;

  tensor::kernels::cpu::ReLU(this->getData(), result.getData(),
                             backwardsMask.getData(), nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradReLU>(backwardsMask, nextFunctions));
  }
  return result;
}

TensorView TensorView::broadcast(int pos, int dim) {
  if (pos < 0 || pos > static_cast<int>(dimensions.size())) {
    throw std::invalid_argument("broadcast pos out of range");
  }
  std::vector<int> newDims = dimensions;
  newDims.insert(newDims.begin() + pos, dim);

  TensorView result(newDims, false);

  // Calculate product of dims from pos to end
  int blockSize = 1;
  for (int i = pos; i < rank; i++) {
    blockSize *= dimensions[i];
  }

  // Calculate product of dims before pos
  int numBlocks = 1;
  for (int i = 0; i < pos; i++) {
    numBlocks *= dimensions[i];
  }

  tensor::kernels::cpu::broadcast(this->getData(), result.getData(), blockSize,
                                  numBlocks, dim);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    result.setGradFunction(
        std::make_shared<function::GradBroadcast>(*this, pos, nextFunctions));
  }
  return result;
}

TensorView TensorView::tanh() {
  TensorView result(dimensions, false);

  tensor::kernels::cpu::tanh(this->getData(), result.getData(), nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradTanh>(*this, nextFunctions));
  }

  return result;
}
TensorView TensorView::exponential() {
  TensorView result(dimensions, false);

  tensor::kernels::cpu::exponential(this->getData(), result.getData(), nValues);

  if (!this->requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradExponential>(*this, nextFunctions));
  }

  return result;
}

TensorView& TensorView::operator+=(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise addition");
  }

  TensorView oldThis = *this;

  tensor::kernels::cpu::elementwiseAdd(this->getData(), other.getData(),
                                       this->getData(), this->nValues);

  if (this->requiresGrad || other.requiresGrad) {
    std::vector<TensorView> savedTensors{oldThis, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradAdd>(savedTensors, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator-=(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise subtraction");
  }

  TensorView oldThis = *this;

  tensor::kernels::cpu::elementwiseSubtract(this->getData(), other.getData(),
                                            this->getData(), this->nValues);

  if (this->requiresGrad || other.requiresGrad) {
    std::vector<TensorView> savedTensors{oldThis, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradSubtract>(savedTensors, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator/=(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise division");
  }

  TensorView oldThis = this->deepCopy();

  tensor::kernels::cpu::elementwiseDivision(this->getData(), other.getData(),
                                            this->getData(), this->nValues);

  if (this->requiresGrad || other.requiresGrad) {
    std::vector<TensorView> savedTensors{oldThis, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradDivide>(savedTensors, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator*=(const TensorView& other) {
  if (dimensions != other.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise multiplication");
  }

  TensorView oldThis = this->deepCopy();

  tensor::kernels::cpu::elementwiseMultiplication(
      this->getData(), other.getData(), this->getData(), this->nValues);

  if (this->requiresGrad || other.requiresGrad) {
    std::vector<TensorView> savedTensors{oldThis, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradMultiply>(savedTensors, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator+=(double scalar) {
  TensorView oldThis = *this;

  tensor::kernels::cpu::tensorScalarAdd(this->getData(), scalar,
                                        this->getData(), this->nValues);

  if (this->requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradAddScalar>(scalar, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator-=(double scalar) {
  TensorView oldThis = *this;

  tensor::kernels::cpu::tensorScalarSubtract(this->getData(), scalar,
                                             this->getData(), this->nValues);

  if (this->requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradSubtractScalar>(scalar, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator*=(double scalar) {
  TensorView oldThis = *this;

  tensor::kernels::cpu::tensorScalarMultiplication(
      this->getData(), scalar, this->getData(), this->nValues);

  if (this->requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradMultiplyScalar>(scalar, nextFunctions));
  }

  return *this;
}

TensorView& TensorView::operator/=(double scalar) {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }

  TensorView oldThis = *this;

  tensor::kernels::cpu::tensorScalarDivision(this->getData(), scalar,
                                             this->getData(), this->nValues);

  if (this->requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(std::make_shared<function::GradDivideScalar>(
        scalar, std::make_shared<TensorView>(oldThis), nextFunctions, true));
  }

  return *this;
}
// Checks if the values within each of the tensors are within the tolerance of
// 1e-9 of being equal to each other in every element
bool TensorView::operator==(const TensorView& other) const {
  if (dimensions != other.dimensions) {
    return false;
  }

  const double epsilon{1e-9};
  for (int i{0}; i < nValues; i++) {
    if (std::abs(getValueDirect(i) - other.getValueDirect(i)) > epsilon) {
      return false;
    }
  }

  return true;
}

// Checks if the values within each of the tensors arent within the tolerance
// of 1e-9 of being equal to each other
bool TensorView::operator!=(const TensorView& other) const {
  return !(*this == other);
}
}  // namespace mattTorch
