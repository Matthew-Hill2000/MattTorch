#include <mattTorch/mattTorch.h>

#include <iostream>
#include <print>

int main(int argc, char* argv[]) {
  // You can perform a large PLETHORA of built in mathematical operations on or
  // between tensors including:
  //
  // - Elementwise multiplication
  // - Elementwise Division
  // - Elementwise Addition
  // - Elementwise subtraction
  //
  // - Scalar multiplication
  // - Scalar Division
  // - Scalar Addition
  // - Scalar subtraction
  //
  // - all including inplace versions (i.e +=)
  //
  // ------------------------------------
  //
  // - Matrix Multiplication
  // - Matrix Transpose Multiplication (i.e A^T*B and A*B^T)
  //
  // ------------------------------------
  //
  // - Scalar Elementwise Exponentiation (i.e A^b)
  //
  // ------------------------------------
  //
  // - Dimensional Broadcast
  // - Dimensional Reduction
  //
  // ------------------------------------
  //
  // - Elementwise Natural Logarithm
  // - Elementwise Tanh
  // - Elementwise ReLU
  // - ELementwise Exponential (i.e e^A)
  //
  // ------------------------------------
  //
  //  - Full Mean
  //
  //

  // Once you understand how to create and manipulate the elements of a Tensor
  // object manually, it is important to understand what mathematical operations
  // can be applied to tensors and how they are applied.

  mattTorch::Tensor smallTensorOne({2, 2});
  mattTorch::Tensor smallTensorTwo({2, 2});
  mattTorch::Tensor largeTensor({3, 3, 3});

  double scalar{4.0};

  for (int i{0}; i < smallTensorOne.getDimensions()[0]; i++) {
    for (int j{0}; j < smallTensorOne.getDimensions()[1]; j++) {
      smallTensorOne[{i, j}] = (i * j) + 1;
      smallTensorTwo[{i, j}] = (2 * i) + (3 * j) + 2;
    }
  }

  for (int i{0}; i < largeTensor.getDimensions()[0]; i++) {
    for (int j{0}; j < largeTensor.getDimensions()[1]; j++) {
      for (int k{0}; k < largeTensor.getDimensions()[2]; k++) {
        largeTensor[{i, j, k}] = i + j + k + 3;
      }
    }
  }
  std::println("smallTensorOne: ");
  std::cout << smallTensorOne << '\n';
  std::println("smallTensorTwo: ");
  std::cout << smallTensorTwo << '\n';

  std::println("largeTensor: ");
  std::cout << largeTensor << '\n';

  // The + operator is overloaded to perform elementwise addition between two
  // tensors of equal dimensions. The operator begins by initialising a tensor
  // with dimensions equal to those of the operands. The pointers to the data
  // within each of the operands and the result tensor are then passed to the
  // elementwiseAdd kernel, which is designed to carry out the additions
  // across each element and store the results as efficiently as possible.
  //
  // The elementwiseAdd kernel processes 8 doubles from the lhs
  // Tensor, 8 doubles from rhs, and produces 8 doubles in result. Since 1
  // double is 8 bytes, 8 doubles take up 64 bytes. This means that each loop
  // iteration within the SIMD loop touches 64B from lhs, 64B from rhs, and
  // writes 64B to result using two AVX2 vector operations.
  //
  // When running through a program, the CPU never actually fetches individual
  // values on their own, instead choosing to fetch cache lines. The idea behind
  // this is that loops often access values in memory that are contiguous, and
  // if your loops dont, then you should probably think about if and how you can
  // make that the case. So in expectation of a nice sequential memory access
  // pattern, the CPU fetches an entire cache line from memory such that the
  // values needed for subsequent iterations of the loop are already loaded into
  // cache. The cache lines are typically 64B long. Since each iteration
  // processes 8 doubles (64B) from both lhs and rhs, an iteration consumes one
  // cache line from each input array when the data is suitably aligned.
  //
  // Remember that when a Tensor object is created, we specified that the memory
  // allocated should be aligned to a memory address that is a multiple of 64.
  // This ensures that the data is nicely aligned for cache line loads. In
  // practice, 64B alignment guarantees that each AVX2 load is also aligned for
  // a 256-bit register and reduces the likelihood that loads cross cache line
  // boundaries, avoiding split-load penalties.
  //
  // On the first iteration of the loop there is not yet any data loaded into
  // cache. The CPU checks the L1 cache, this results in a miss and a subsequent
  // check of L2, then a check of L3, and finally a check into DRAM. The cache
  // lines holding lhs and rhs are then fetched from DRAM and filled into L3,
  // then L2, and finally into L1D where the loads can be satisfied.
  //
  // Importantly, each miss triggers a full 64B cache line transfer, not just
  // the bytes that the SIMD instructions immediately consume. This means that a
  // single cache line fetch can provide data for future iterations, assuming a
  // contiguous access pattern. Hardware prefetchers will typically detect this
  // linear stride pattern (i += 8) and begin speculatively loading subsequent
  // cache lines into L2/L1 before they are explicitly requested by the load
  // instructions.
  //
  // Once the data is within L1D, subsequent iterations no longer require L3 or
  // DRAM accesses, instead loading from L1D so long as the data remains present
  // there, with much lower latency (typically ~4 cycles for L1D, ~10-15 cycles
  // for L2, ~30-60 cycles for L3, and hundreds of cycles for DRAM). The loaded
  // values are then moved into SIMD registers (YMM registers for AVX2), where
  // the actual arithmetic operations are performed.
  //
  // The loads:
  //
  //   __m256d a0 = _mm256_load_pd(lhs + i);
  //   __m256d b0 = _mm256_load_pd(rhs + i);
  //   __m256d a1 = _mm256_load_pd(lhs + i + 4);
  //   __m256d b1 = _mm256_load_pd(rhs + i + 4);
  //
  // bring 8 doubles from each input array into two pairs of 256-bit vector
  // registers. The addition operations are executed on the floating-point SIMD
  // execution units, which operate on all 4 lanes of each vector in parallel.
  //
  // The result vectors are then written back using non-temporal (streaming)
  // stores via _mm256_stream_pd. These stores are intended for write-only
  // output streams and avoid the usual write-allocate behaviour that would
  // otherwise require a Read For Ownership (RFO) fetch of the destination cache
  // line. The stores are buffered by the CPU and written through the memory
  // hierarchy with reduced cache pollution compared to ordinary stores.
  //
  // Eventually, the buffered streaming stores are committed through the memory
  // hierarchy and written back to memory. However, in the steady-state
  // streaming case, memory bandwidth rather than compute throughput typically
  // becomes the dominant performance constraint.
  //
  // The SIMD utilising loop of the elementwiseAdd kernel operates for all
  // values of i such that i+7 < nValues. This ensures that there are at least 8
  // values remaining for us to fill our vector registers. Once this is no
  // longer the case, a scalar loop iterates over the final values of i, one at
  // a time, calculating and storing the result with normal single scalar
  // addition.
  //
  // Once the elementwiseAdd kernel has finished its job, the resulting Tensor
  // is returned by value from the + overload.
  //
  // The Tensor class has its move constructor and assignment operator
  // defaulted. This means that the resulting tensor is moved into the call site
  // rather than copied. In practice, the compiler usually should utilise NRVO,
  // where it constructs the resulting Tensor at the call site from the start,
  // with the foresight that it will be returned eventually.

  std::println("-----------------------------------------------------------");

  std::println("Tensor + Tensor \n");
  auto r1 = smallTensorOne + smallTensorTwo;
  std::cout << r1 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor - Tensor \n");
  auto r2 = smallTensorOne - smallTensorTwo;
  std::cout << r2 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor * Tensor (elementwise) \n");
  auto r3 = smallTensorOne * smallTensorTwo;
  std::cout << r3 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor / Tensor (elementwise) \n");
  auto r4 = smallTensorOne / smallTensorTwo;
  std::cout << r4 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor + scalar \n");
  auto r5 = smallTensorOne + 1.0;
  std::cout << r5 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor - scalar \n");
  auto r6 = smallTensorOne - 1.0;
  std::cout << r6 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor * scalar \n");
  auto r7 = smallTensorOne * 2.0;
  std::cout << r7 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor / scalar \n");
  auto r8 = smallTensorOne / 2.0;
  std::cout << r8 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor elementwise exponent \n");
  auto r9 = smallTensorOne.elementwiseExponent(2);
  std::cout << r9 << "\n";
  std::println("-----------------------------------------------------------");

  // reductionSum performs a dimensional reduction over a specified index of an
  // Tensor. This operation reduces an entire dimension by accumulating values
  // along that dimension.
  //
  // If we choose a dimension `dim`, the tensor is logically split into three
  // parts:
  //
  //   - outerSize: the number of independent "blocks" before the reduced axis
  //   - reduceSize: the length of the axis being summed over
  //   - innerSize: the number of elements after the reduced axis
  //
  // This effectively reinterprets a flattened memory layout as:
  //
  //   [ outer ][ reduce ][ inner ]
  //
  // where elements along the "reduce" dimension are accessed in a strided
  // pattern defined by innerSize.
  //
  // For each (outer, inner) pair, the kernel computes a single scalar output by
  // summing across the entire reduce dimension:
  //
  //   acc = input[outer][0][inner]
  //       + input[outer][1][inner]
  //       + ...
  //       + input[outer][reduceSize - 1][inner]
  //
  // This means the reduction collapses one dimension, producing an output
  // tensor with one fewer dimension.
  //
  // In memory terms, the input is stored as a flat contiguous array, so the
  // kernel computes the correct index manually:
  //
  //   idx = outer * reduceSize * innerSize
  //       + r * innerSize
  //       + inner
  //
  // This stride pattern is important: the `innerSize` term determines how far
  // apart elements are along the reduced dimension, meaning each accumulation
  // step jumps through memory with a fixed stride rather than being fully
  // contiguous.
  //

  std::println("Tensor reduction sum \n");
  auto r10 = largeTensor.reductionSum(1);
  std::cout << r10 << "\n";
  std::println("-----------------------------------------------------------");

  // broadcast takes a lower-rank tensor and expands it along a newly inserted
  // dimension without actually copying or modifying the logical values of the
  // original tensor. Instead, it constructs a new shape where values are
  // repeated across a dimension, while the underlying data is replicated in
  // memory.
  //
  // A new dimension of size `dim` is inserted at position `pos`,
  // transforming the tensor shape from:
  //
  //   [ ... old dims ... ]
  //
  // into:
  //
  //   [ ... before pos ... , dim , ... after pos ... ]
  //
  // This does not change the underlying values; it only changes how many times
  // those values are repeated in the output layout.
  //
  // To implement this efficiently in a flattened memory representation, the
  // tensor is viewed as a sequence of:
  //
  //   numBlocks * blockSize
  //
  // where:
  //
  //   - blockSize = product of the original dimensions from `pos` onwards
  //   - numBlocks = product of the original dimensions before `pos`
  //
  // From the perspective of the result, blockSize is the size of the
  // contiguous chunk that sits after the inserted dimension, and numBlocks
  // is how many of those chunks the inserted dimension partitions the tensor
  // into.
  //
  // Each "block" corresponds to a contiguous region of memory that should be
  // replicated independently.
  //
  // Within each block, the kernel performs a repeated memory copy:
  //
  //   for each block:
  //       repeat `repeatCount` times:
  //           copy blockSize elements into the correct offset in output
  //
  // This is implemented using memcpy:
  //
  //   std::memcpy(destination, source, blockSize * sizeof(double))
  //
  // meaning each broadcast operation copies a contiguous chunk of memory rather
  // than performing elementwise work in a loop.

  std::println("Tensor broadcast \n");
  auto r16 = smallTensorOne.broadcast(0, 3);
  std::cout << r16 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor mean \n");
  auto r11 = smallTensorOne.mean();
  std::cout << r11 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor log \n");
  auto r12 = smallTensorOne.log();
  std::cout << r12 << "\n";
  std::println("-----------------------------------------------------------");

  // The matrixMultiply method performs matrix multiplication between two rank-2
  // Tensors, computing C = A * B where A has dimensions (lhsRows x lhsCols) and
  // B has dimensions (lhsCols x rhsCols), producing a result of dimensions
  // (lhsRows x rhsCols). The method first verifies that both tensors are of
  // rank 2 and that the inner dimensions agree, before constructing a
  // zero-initialised result Tensor and calling the
  // matrixMultiplicationMegaKernel.
  //
  // To appreciate why a MegaKernel is needed at all, consider the naive
  // triple-loop implementation of matrix multiplication:
  //
  //   for i in 0..lhsRows:
  //     for j in 0..rhsCols:
  //       for k in 0..lhsCols:
  //         C[i][j] += A[i][k] * B[k][j]
  //
  // In this ijk ordering the innermost loop increments k, which corresponds to
  // moving along a single row of A (stride 1 in memory, good) but down a single
  // column of B (stride rhsCols in memory, bad). For large matrices, each
  // access to B[k][j] steps by rhsCols doubles in the underlying contiguous
  // storage, meaning consecutive iterations touch entirely different cache
  // lines. Each load triggers a fresh cache miss, forcing a stream of L3 or
  // DRAM fetches. For matrices much larger than the L3 cache this access
  // pattern becomes bandwidth limited by DRAM and achieves only a tiny fraction
  // of the CPU's peak floating-point throughput.
  //
  // The matrixMultiplicationMegaKernel addresses this with three complementary
  // techniques: cache blocking, data packing, and SIMD register tiling, all
  // combined under by OpenMP multi-threading.
  //
  // Cache blocking partitions A and B into square sub-tiles of size
  // BIGBLOCKSIZE x BIGBLOCKSIZE (currently 128 x 128 doubles, i.e. 128*128*8
  // = 131072 bytes per tile). For a typical CPU with an L2 cache of 256 KiB to
  // 512 KiB per core, holding two such tiles simultaneously keeps the working
  // set comfortably within L2, so that the innermost arithmetic operations
  // encounter data that is already hot in cache. Without blocking the working
  // set of the inner loops spans entire rows or columns of the full matrix,
  // which can be far too large to fit in L2, causing repeated L3 or DRAM
  // fetches for values that have already been loaded once.
  //
  // The loop structure in matrixMultiplicationMegaKernel iterates over three
  // block-level indices: rhsCol (blocks across the columns of B and the
  // result), lhsRow (blocks down the rows of A and the result), and lhsCol
  // (blocks along the shared inner dimension):
  //
  //   for rhsCol in 0..rhsCols step BIGBLOCKSIZE_J:   // parallelised
  //     for lhsRow in 0..lhsRows step BIGBLOCKSIZE_I: // parallelised
  //       for lhsCol in 0..lhsCols step BIGBLOCKSIZE_K:
  //         pack rhs block into rhsBigBlock
  //         pack lhs block into lhsBigBlock
  //         call matrixMultiplicationMicroKernel
  //
  // The outermost two loops are parallelised using OpenMP with collapse(2) and
  // dynamic scheduling. collapse(2) causes the OpenMP runtime to treat the
  // two-dimensional iteration space as a single flat pool of work items, each
  // corresponding to one (rhsCol, lhsRow) block pair. dynamic scheduling then
  // assigns these work items to threads on demand as threads become free,
  // rather than dividing the work statically upfront. This is beneficial when
  // block computation times are uneven, for example when the last block along a
  // dimension is smaller than BIGBLOCKSIZE because the matrix dimensions are
  // not exact multiples. Each thread allocates its own private rhsBigBlock and
  // lhsBigBlock buffers using posix_memalign, so threads never contend over the
  // same pack buffers.
  //
  // Before the micro-kernel can run, the current block of both A and B is
  // repacked into a dense, cache-friendly layout by rhsBigPack and lhsBigPack
  // respectively. This step is critical: even after cache blocking ensures that
  // the block fits in L2, the original row-major layout of both matrices still
  // causes the inner loops to access data with non-unit strides when computing
  // individual dot products. Packing transforms the blocks into a layout that
  // allows the micro-kernel to execute with entirely unit-stride loads
  // throughout.
  //
  // rhsBigPack reorganises the current (kMax x jMax) block of B so that for
  // each group of 8 contiguous columns (an Nr=8 panel), the kMax values along
  // the shared inner dimension for all 8 columns are interleaved contiguously
  // in memory. Specifically, for a panel starting at column j, the packed
  // layout is:
  //
  //   rhsBigBlock[j * kMax + 8 * k + 0..7] = B[lhsCol+k][rhsCol+j .. j+7]
  //
  // meaning for each k, the 8 column values for that row are stored
  // sequentially at offset j*kMax + 8*k within the buffer. When the
  // micro-kernel iterates over k, each step loads 8 consecutive doubles from a
  // single contiguous address, which maps onto two 256-bit AVX2 loads with no
  // wasted bytes. Intermediate panels of width 4 (when 4 to 7 columns remain
  // after the 8-wide panels) are packed similarly but with groups of 4, and
  // any scalar tail columns are packed with a stride of kMax.
  //
  // lhsBigPack reorganises the current (iMax x kMax) block of A so that for
  // each group of 6 contiguous rows (the Mr=6 height of the micro-kernel), the
  // kMax values along the shared inner dimension for all 6 rows are interleaved
  // contiguously. For a row group starting at row i:
  //
  //   lhsBigBlock[i * kMax + 6 * k + 0..5] = A[lhsRow+i .. i+5][lhsCol+k]
  //
  // meaning for each k, the 6 row values at that column position are stored
  // sequentially. When the micro-kernel broadcasts a scalar A[i][k] into an
  // AVX2 register, it loads from lhsBigBlock[i*kMax + 6*k + row_offset], which
  // is always a unit-stride access into the packed buffer.
  //
  // The micro-kernel matrixMultiplicationMicroKernel operates on a 6 x 8
  // register tile of the result. For each group of 6 result rows (i loop in
  // steps of 6) and each group of 8 result columns (j loop in steps of 8), it
  // maintains 12 AVX2 accumulator registers in the YMM register file:
  //
  //   c_i0 holds C[lhsRow+i][rhsCol+j .. j+3]   for i in 0..5
  //   c_i4 holds C[lhsRow+i][rhsCol+j+4 .. j+7] for i in 0..5
  //
  // These 12 accumulators are first loaded from the existing values in the
  // result matrix, since the outer lhsCol loop accumulates partial sums from
  // successive k-blocks. The inner k loop then performs the actual arithmetic:
  // for each k in 0..kMax, two 256-bit vectors b0 and b1 are loaded from the
  // packed rhs buffer, covering the 8 column values for that k:
  //
  //   __m256d b0 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 0]);
  //   __m256d b1 = _mm256_loadu_pd(&rhsBlock[kMax * j + 8 * k + 4]);
  //
  // Six scalar lhs values are then each broadcast into a full 256-bit register:
  //
  //   __m256d a0 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 0]);
  //   ...
  //   __m256d a5 = _mm256_broadcast_sd(&lhsBlock[kMax * i + 6 * k + 5]);
  //
  // producing six vectors of the form (A[i][k] A[i][k] A[i][k] A[i][k]). Each
  // of the 12 accumulators is then updated with a fused multiply-add:
  //
  //   acc_i0 = _mm256_fmadd_pd(a_i, b0, acc_i0)
  //   acc_i1 = _mm256_fmadd_pd(a_i, b1, acc_i1)
  //
  // This issues 12 FMA operations per k step, each computing 4 double-precision
  // fused multiply-adds in a single instruction. The FMA operation computes
  // a * b + c in a single pass, avoiding the rounding error of a separate
  // multiply followed by an add and delivering twice the floating-point
  // throughput of separate multiply and add instructions on CPUs that can issue
  // one FMA per clock cycle per execution unit.
  //
  // After all kMax steps, the 12 accumulators hold the updated partial sums for
  // the 6 x 8 tile and are written back to the result matrix using unaligned
  // stores. The j loop also has intermediate fallback paths: a 6 x 4 tile
  // handles 4 to 7 remaining columns using 6 accumulators and a single b
  // vector, and a scalar j-tail handles any remaining columns one at a time.
  // An i-tail similarly handles rows that are not a multiple of 6, mirroring
  // the j-layer structure but operating on a single row at a time.
  //
  // The overall effect is a kernel that sustains a high fraction of the
  // hardware's peak FMA throughput: the packed data layouts ensure that all
  // inner-loop loads are unit-stride and fully utilise AVX2 load bandwidth, the
  // 6 x 8 register tile keeps 12 YMM accumulators resident in the register
  // file throughout the inner k loop, and the three-level blocking keeps the
  // operand data within L2 cache for the duration of each block computation.

  std::println("Tensor matrix multiply \n");
  auto r13 = smallTensorOne.matrixMultiply(smallTensorTwo);
  std::cout << r13 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor transpose multiply \n");
  auto r14 = smallTensorOne.transposeMultiply(smallTensorTwo, true);
  std::cout << r14 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor ReLU \n");
  auto r15 = smallTensorOne.ReLU();
  std::cout << r15 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor tanh \n");
  auto r17 = smallTensorOne.tanh();
  std::cout << r17 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor exponential \n");
  auto r18 = smallTensorOne.exponential();
  std::cout << r18 << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor += Tensor \n");
  smallTensorOne += smallTensorTwo;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor -= Tensor \n");
  smallTensorOne -= smallTensorTwo;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor *= Tensor \n");
  smallTensorOne *= smallTensorTwo;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor /= Tensor \n");
  smallTensorOne /= smallTensorTwo;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor += scalar \n");
  smallTensorOne += 1.0;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor -= scalar \n");
  smallTensorOne -= 1.0;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor *= scalar \n");
  smallTensorOne *= 2.0;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor /= scalar \n");
  smallTensorOne /= 2.0;
  std::cout << smallTensorOne << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor == Tensor \n");
  bool eq = (smallTensorOne == smallTensorTwo);
  std::cout << eq << "\n";
  std::println("-----------------------------------------------------------");

  std::println("Tensor != Tensor \n");
  bool neq = (smallTensorOne != smallTensorTwo);
  std::cout << neq << "\n";
  std::println("-----------------------------------------------------------");
}
