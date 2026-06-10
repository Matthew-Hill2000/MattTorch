#pragma once

#include <mattTorch/function/gradFunction.h>

// Accumulator
#include <mattTorch/function/accumulator/gradAccumulator.h>

// Addition
#include <mattTorch/function/addition/gradAdd.h>
#include <mattTorch/function/addition/gradAddScalar.h>

// Subtraction
#include <mattTorch/function/subtraction/gradSubtract.h>
#include <mattTorch/function/subtraction/gradSubtractScalar.h>

// Multiplication
#include <mattTorch/function/multiplication/gradMultiply.h>
#include <mattTorch/function/multiplication/gradMultiplyMatrix.h>
#include <mattTorch/function/multiplication/gradMultiplyScalar.h>
#include <mattTorch/function/multiplication/gradTransposeMatrix.h>

// Division
#include <mattTorch/function/division/gradDivide.h>
#include <mattTorch/function/division/gradDivideScalar.h>

// Exponent
#include <mattTorch/function/exponent/gradExponent.h>

// ReLU
#include <mattTorch/function/ReLU/gradReLU.h>

// Reduction Sum
#include <mattTorch/function/reductionSum/gradReductionSum.h>

// Broadcast
#include <mattTorch/function/broadcast/gradBroadcast.h>

// Mean
#include <mattTorch/function/mean/gradMean.h>

// Exponential
#include <mattTorch/function/exponential/gradExponential.h>

// Log
#include <mattTorch/function/log/gradLog.h>

// Tanh
#include <mattTorch/function/tanh/gradTanh.h>
