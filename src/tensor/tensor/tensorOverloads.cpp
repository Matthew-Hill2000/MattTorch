#include <mattTorch/function/functionInclude.h>
#include <mattTorch/tensor/kernels/kernels.h>
#include <mattTorch/tensor/tensor/tensor.h>

#include <cmath>
#include <memory>
#include <stdexcept>
#include <utility>

namespace mattTorch {

Tensor& Tensor::operator=(double val) {
  this->storage->setAllValues(val);
  return *this;
}

Tensor Tensor::operator+(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument("Tensor dimensions must match for addition");
  }

  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::elementwiseAdd(this->getData(), other.getData(),
                                       result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad && !other.gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(std::make_shared<function::GradAdd>(nextFunctions));
  }

  return result;
}

Tensor Tensor::operator-(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument("Tensor dimensions must match for subtraction");
  }

  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::elementwiseSubtract(
      this->getData(), other.getData(), result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad && !other.gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradSubtract>(nextFunctions));
  }

  return result;
}

Tensor Tensor::operator*(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise multiplication");
  }

  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::elementwiseMultiplication(
      this->getData(), other.getData(), result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad && !other.gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<Tensor> savedTensors{*this, other};

    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradMultiply>(savedTensors, nextFunctions));
  }
  return result;
}

Tensor Tensor::operator/(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument("Tensor dimensions must match for division");
  }

  Tensor result(tensorData.dimensions, false);

  double* divisor = other.getData();
  for (int i{0}; i < getNValues(); i++) {
    if (divisor[i] == 0.0) {
      throw std::invalid_argument("Division by zero");
    }
  }

  tensor::kernels::cpu::elementwiseDivision(
      this->getData(), other.getData(), result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad && !other.gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<Tensor> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradDivide>(savedTensors, nextFunctions));
  }

  return result;
}

Tensor Tensor::operator+(double scalar) {
  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::tensorScalarAdd(this->getData(), scalar,
                                        result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradAddScalar>(scalar, nextFunctions));
  }

  return result;
}

Tensor Tensor::operator-(double scalar) {
  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::tensorScalarSubtract(
      this->getData(), scalar, result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradSubtractScalar>(scalar, nextFunctions));
  }
  return result;
}

Tensor Tensor::operator*(double scalar) {
  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::tensorScalarMultiplication(
      this->getData(), scalar, result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradMultiplyScalar>(scalar, nextFunctions));
  }
  return result;
}

Tensor Tensor::operator/(double scalar) {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }

  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::tensorScalarDivision(
      this->getData(), scalar, result.getData(), this->getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(std::make_shared<function::GradDivideScalar>(
        scalar, std::make_shared<Tensor>(*this), nextFunctions, true));
  }
  return result;
}

Tensor Tensor::elementwiseExponent(int scalar) {
  if (scalar == 0.0) {
    return *this;
  }

  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::elementwiseExponent(this->getData(), scalar,
                                            result.getData(), getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<int> savedScalars{scalar};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    std::vector<Tensor> savedTensors{*this};

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(std::make_shared<function::GradExponent>(
        savedScalars, savedTensors, nextFunctions));
  }
  return result;
}

Tensor Tensor::reductionSum(int dim) {
  const std::size_t rank = getRank();
  if (dim < 0 || std::cmp_greater_equal(dim, rank)) {
    throw std::invalid_argument("Invalid dimension for reduction");
  }

  // Build new dimensions with the reduced dim removed
  Dims newDimensions;
  for (std::size_t i = 0; i < rank; i++) {
    if (std::cmp_not_equal(i, dim)) {
      newDimensions.push_back(tensorData.dimensions[i]);
    }
  }
  if (newDimensions.empty()) {
    newDimensions.push_back(1);
  }

  Tensor result(newDimensions, false);

  // Calculate strides for the kernel
  int outerSize = 1;
  for (int i = 0; i < dim; i++) {
    outerSize *= tensorData.dimensions[i];
  }

  int reduceSize = tensorData.dimensions[dim];

  int innerSize = 1;
  for (std::size_t i = static_cast<std::size_t>(dim) + 1; i < rank; i++) {
    innerSize *= tensorData.dimensions[i];
  }

  tensor::kernels::cpu::reductionSum(this->getData(), result.getData(),
                                     outerSize, reduceSize, innerSize);

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<Tensor> savedTensors{*this};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    result.setGradFunction(std::make_shared<function::GradReductionSum>(
        savedTensors, dim, nextFunctions));
  }
  return result;
}

Tensor Tensor::mean() {
  Tensor result({1});

  tensor::kernels::cpu::mean(this->getData(), result.getData(), getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradMean>(*this, nextFunctions));
  }
  return result;
}

Tensor Tensor::log() {
  Tensor result(tensorData.dimensions);

  tensor::kernels::cpu::log(this->getData(), result.getData(), getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradLog>(*this, nextFunctions));
  }
  return result;
}

Tensor Tensor::matrixMultiply(const Tensor& other) {
  if (tensorData.dimensions.size() != 2 ||
      other.tensorData.dimensions.size() != 2) {
    throw std::invalid_argument("Tensors should be of rank 2");
  }

  if (tensorData.dimensions[1] != other.tensorData.dimensions[0]) {
    throw std::invalid_argument("Tensors innermost dimensions should match");
  }

  Tensor result(Dims{tensorData.dimensions[0], other.tensorData.dimensions[1]},
                false);

  tensor::kernels::cpu::matrixMultBlockPackedVector(
      this->getData(), other.getData(), result.getData(),
      this->tensorData.dimensions[0], this->tensorData.dimensions[1],
      other.tensorData.dimensions[1]);

  if (!this->gradData.requiresGrad && !other.gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<Tensor> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(std::make_shared<function::GradMultiplyMatrix>(
        savedTensors, nextFunctions));
  }
  return result;
}

Tensor Tensor::transposeMultiply(const Tensor& other, bool transposeFirst) {
  if (tensorData.dimensions.size() != 2 ||
      other.tensorData.dimensions.size() != 2) {
    throw std::invalid_argument("Tensors should be of rank 2");
  }

  const std::size_t dimIndex = transposeFirst ? 0 : 1;

  if (tensorData.dimensions[dimIndex] !=
      other.tensorData.dimensions[dimIndex]) {
    throw std::invalid_argument("Tensors innermost dimensions should match");
  }

  Tensor result;

  result =
      Tensor(Dims{tensorData.dimensions[transposeFirst == true ? 1 : 0],
                  other.tensorData.dimensions[transposeFirst == true ? 1 : 0]},
             false);

  if (transposeFirst) {
    tensor::kernels::cpu::transposeMultBlockVectorLHS(
        this->getData(), other.getData(), result.getData(),
        this->tensorData.dimensions[0], this->tensorData.dimensions[1],
        other.tensorData.dimensions[1]);

  } else {
    tensor::kernels::cpu::transposeMultBlockVectorRHS(
        this->getData(), other.getData(), result.getData(),
        this->tensorData.dimensions[0], this->tensorData.dimensions[1],
        other.tensorData.dimensions[0]);
  }

  if (!this->gradData.requiresGrad && !other.gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<Tensor> savedTensors{*this, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);

    result.setGradFunction(std::make_shared<function::GradTransposeMatrix>(
        savedTensors, nextFunctions, transposeFirst == true ? true : false));
  }
  return result;
}

Tensor Tensor::ReLU() {
  Tensor result(tensorData.dimensions, false);
  Tensor backwardsMask(tensorData.dimensions, false);
  backwardsMask.setRequiresGrad(false);
  backwardsMask = 1.0;

  tensor::kernels::cpu::ReLU(this->getData(), result.getData(),
                             backwardsMask.getData(), getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradReLU>(backwardsMask, nextFunctions));
  }
  return result;
}

Tensor Tensor::broadcast(int pos, int dim) {
  if (pos < 0 || static_cast<std::size_t>(pos) > tensorData.dimensions.size()) {
    throw std::invalid_argument("broadcast pos out of range");
  }
  Dims newDims = tensorData.dimensions;
  newDims.insert(newDims.begin() + pos, dim);

  Tensor result(newDims, false);

  int blockSize = 1;
  for (auto i = static_cast<std::size_t>(pos); i < getRank(); i++) {
    blockSize *= tensorData.dimensions[i];
  }

  int numBlocks = 1;
  for (int i = 0; i < pos; i++) {
    numBlocks *= tensorData.dimensions[i];
  }

  tensor::kernels::cpu::broadcast(this->getData(), result.getData(), blockSize,
                                  numBlocks, dim);

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    result.setGradFunction(
        std::make_shared<function::GradBroadcast>(*this, pos, nextFunctions));
  }
  return result;
}

Tensor Tensor::tanh() {
  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::tanh(this->getData(), result.getData(), getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradTanh>(*this, nextFunctions));
  }

  return result;
}
Tensor Tensor::exponential() {
  Tensor result(tensorData.dimensions, false);

  tensor::kernels::cpu::exponential(this->getData(), result.getData(),
                                    getNValues());

  if (!this->gradData.requiresGrad) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);

    result.setGradFunction(
        std::make_shared<function::GradExponential>(*this, nextFunctions));
  }

  return result;
}

Tensor& Tensor::operator+=(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise addition!");
  }

  tensor::kernels::cpu::elementwiseAdd(this->getData(), other.getData(),
                                       this->getData(), this->getNValues());

  if (this->gradData.requiresGrad || other.gradData.requiresGrad) {
    this->setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(std::make_shared<function::GradAdd>(nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator-=(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise subtraction");
  }

  tensor::kernels::cpu::elementwiseSubtract(
      this->getData(), other.getData(), this->getData(), this->getNValues());

  if (this->gradData.requiresGrad || other.gradData.requiresGrad) {
    this->setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradSubtract>(nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator/=(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise division");
  }

  Tensor oldThis = this->deepCopy();

  tensor::kernels::cpu::elementwiseDivision(
      this->getData(), other.getData(), this->getData(), this->getNValues());

  if (this->gradData.requiresGrad || other.gradData.requiresGrad) {
    this->setRequiresGrad(true);
    std::vector<Tensor> savedTensors{oldThis, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradDivide>(savedTensors, nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator*=(const Tensor& other) {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    throw std::invalid_argument(
        "Tensor dimensions must match for element-wise multiplication");
  }

  Tensor oldThis = this->deepCopy();

  tensor::kernels::cpu::elementwiseMultiplication(
      this->getData(), other.getData(), this->getData(), this->getNValues());

  if (this->gradData.requiresGrad || other.gradData.requiresGrad) {
    this->setRequiresGrad(true);
    std::vector<Tensor> savedTensors{oldThis, other};
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    nextFunctions.push_back(other.gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradMultiply>(savedTensors, nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator+=(double scalar) {
  tensor::kernels::cpu::tensorScalarAdd(this->getData(), scalar,
                                        this->getData(), this->getNValues());

  if (this->gradData.requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradAddScalar>(scalar, nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator-=(double scalar) {
  tensor::kernels::cpu::tensorScalarSubtract(
      this->getData(), scalar, this->getData(), this->getNValues());

  if (this->gradData.requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradSubtractScalar>(scalar, nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator*=(double scalar) {
  Tensor oldThis = this->deepCopy();

  tensor::kernels::cpu::tensorScalarMultiplication(
      this->getData(), scalar, this->getData(), this->getNValues());

  if (this->gradData.requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(
        std::make_shared<function::GradMultiplyScalar>(scalar, nextFunctions));
  }

  return *this;
}

Tensor& Tensor::operator/=(double scalar) {
  if (scalar == 0.0) {
    throw std::invalid_argument("Division by zero");
  }

  Tensor oldThis = this->deepCopy();

  tensor::kernels::cpu::tensorScalarDivision(
      this->getData(), scalar, this->getData(), this->getNValues());

  if (this->gradData.requiresGrad) {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;
    nextFunctions.push_back(this->gradFunction);
    this->setGradFunction(std::make_shared<function::GradDivideScalar>(
        scalar, std::make_shared<Tensor>(oldThis), nextFunctions, true));
  }

  return *this;
}

bool Tensor::operator==(const Tensor& other) const {
  if (tensorData.dimensions != other.tensorData.dimensions) {
    return false;
  }

  const double epsilon{1e-9};
  for (int i{0}; i < getNValues(); i++) {
    if (std::abs(getValueDirect(i) - other.getValueDirect(i)) > epsilon) {
      return false;
    }
  }

  return true;
}

bool Tensor::operator!=(const Tensor& other) const { return !(*this == other); }

Tensor operator+(double scalar, Tensor& tensor) {
  Tensor result(tensor.getDimensions(), false);

  tensor::kernels::cpu::tensorScalarAdd(tensor.getData(), scalar,
                                        result.getData(), result.getNValues());
  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(
        std::make_shared<function::GradAddScalar>(scalar, nextFunctions));
  }

  return result;
}

Tensor operator-(double scalar, Tensor& tensor) {
  Tensor result(tensor.getDimensions(), false);

  tensor::kernels::cpu::scalarTensorSubtract(
      tensor.getData(), scalar, result.getData(), result.getNValues());

  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(
        std::make_shared<function::GradSubtractScalar>(scalar, nextFunctions));
  }

  return result;
}

Tensor operator*(double scalar, Tensor& tensor) {
  Tensor result(tensor.getDimensions(), false);

  tensor::kernels::cpu::tensorScalarMultiplication(
      tensor.getData(), scalar, result.getData(), result.getNValues());
  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(
        std::make_shared<function::GradMultiplyScalar>(scalar, nextFunctions));
  }

  return result;
}

Tensor operator/(double scalar, Tensor& tensor) {
  Tensor result(tensor.getDimensions(), false);

  tensor::kernels::cpu::scalarTensorDivision(
      tensor.getData(), scalar, result.getData(), result.getNValues());

  if (!tensor.getRequiresGrad()) {
    result.setRequiresGrad(false);
  } else {
    result.setRequiresGrad(true);
    std::vector<std::shared_ptr<GradFunction>> nextFunctions;

    nextFunctions.push_back(tensor.getGradFunction());

    result.setGradFunction(std::make_shared<function::GradDivideScalar>(
        scalar, std::make_shared<Tensor>(tensor), nextFunctions, false));
  }

  return result;
}
}  // namespace mattTorch
