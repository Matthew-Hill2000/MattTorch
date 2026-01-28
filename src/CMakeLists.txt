add_library(mattTorch)

add_library(mattTorch::mattTorch ALIAS mattTorch)

# Source files
target_sources(mattTorch
    PRIVATE
        function/accumulator/gradAccumulator.cpp

        function/addition/gradAdd.cpp
        function/addition/gradAddScalar.cpp

        function/subtraction/gradSubtract.cpp
        function/subtraction/gradSubtractScalar.cpp

        function/multiplication/gradMultiply.cpp
        function/multiplication/gradMultiplyScalar.cpp
        function/multiplication/gradMultiplyMatrix.cpp
        function/multiplication/gradTransposeMatrix.cpp

        function/division/gradDivide.cpp
        function/division/gradDivideScalar.cpp

        function/exponent/gradExponent.cpp

        function/ReLU/gradReLU.cpp

        function/reductionSum/gradReductionSum.cpp

        function/broadcast/gradBroadcast.cpp
        function/tanh/gradTanh.cpp
        function/log/gradLog.cpp
        function/mean/gradMean.cpp
        function/exponential/gradExponential.cpp

        tensor/kernels/cpu/tensor-tensor/elementwise.cpp
        tensor/kernels/cpu/tensor-tensor/matrix.cpp
        tensor/kernels/cpu/tensor-tensor/matrixVector.cpp
        tensor/kernels/cpu/tensor-tensor/transposeMultLHS.cpp
        tensor/kernels/cpu/tensor-tensor/transposeMultRHS.cpp

        tensor/kernels/cpu/tensor-scalar/tensorScalarElementwise.cpp
        tensor/kernels/cpu/tensor/tensorElementwise.cpp

        tensor/tensorStorage/tensorStorage.cpp

        tensor/tensorView/tensorView.cpp
        tensor/tensorView/tensorViewAccesssors.cpp
        tensor/tensorView/tensorViewBackProp.cpp
        tensor/tensorView/tensorViewOverloads.cpp

        layer/fullyConnectedLayer/fullyConnectedLayer.cpp
        layer/ReLU/ReLU.cpp
        layer/tanh/tanh.cpp
        layer/softmax/softmax.cpp
        
        optimiser/sgd/sgd.cpp
        
        criterion/mseLoss/mseLoss.cpp
        criterion/crossEntropyLoss/crossEntropyLoss.cpp

        network/network.cpp

        dataset/dataset.cpp
)

target_include_directories(mattTorch
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

target_compile_features(mattTorch
    PUBLIC
        cxx_std_23
)

find_package(OpenMP REQUIRED)
target_link_libraries(mattTorch
    PUBLIC
        OpenMP::OpenMP_CXX
)

target_include_directories(mattTorch
    PRIVATE
        /usr/include
)

target_link_libraries(mattTorch
    PRIVATE
        sleef
)
target_compile_options(mattTorch 
    PRIVATE
        -mavx2 -mfma -march=native -fopenmp
)

target_compile_options(mattTorch 
    PRIVATE
        $<$<CONFIG:Release>:-O3 -ffast-math>
        $<$<CONFIG:RelWithDebInfo>:-O2 -ffast-math>
        $<$<CONFIG:Debug>:-O0>
)

target_compile_options(mattTorch
    PRIVATE
        -Wall -Wextra -Wpedantic
)
