#pragma once
#include "simulator.hpp"
namespace sjtu {

void Calculate(std::vector<Matrix *> keys, std::vector<Matrix *> values,
               Rater &rater, GpuSimulator &gpu_sim,
               MatrixMemoryAllocator matrix_memory_allocator) {
  assert(keys.size() == values.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    auto current_query = rater.GetNextQuery();
    /*
     * Implement your calculation logic here.
     * You can use the GpuSimulator instance to perform matrix operations.
     * For example:
     * gpu_sim.MoveMatrixToGpuHbm(keys[i]);
     * When your need a new matrix, to avoid memory leak, you should use
     * Matrix* new_matrix =
     * matrix_memory_allocator.Allocate(YOUR_MATRIX_NAME(string, which is
     * helpful for debugging)); It can manage the memory of matrices
     * automatically.
     */

    // For iteration i, we need to:
    // 1. Concatenate keys[0..i] to form K matrix [i+1, d]
    // 2. Concatenate values[0..i] to form V matrix [i+1, d]
    // 3. Q has shape [i+1, d]
    // 4. Compute Attention(Q, K, V) = Softmax(Q * K^T) * V

    // Step 1: Concatenate all keys into K and all values into V
    // Concatenate vertically (axis=0) since each key/value is [1, d]
    Matrix* K = nullptr;
    Matrix* V = nullptr;

    for (size_t j = 0; j <= i; ++j) {
      if (j == 0) {
        K = matrix_memory_allocator.Allocate("K");
        gpu_sim.Copy(keys[j], K, kInGpuHbm);
        V = matrix_memory_allocator.Allocate("V");
        gpu_sim.Copy(values[j], V, kInGpuHbm);
      } else {
        Matrix* new_K = matrix_memory_allocator.Allocate("new_K");
        gpu_sim.Concat(K, keys[j], new_K, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(K);
        K = new_K;

        Matrix* new_V = matrix_memory_allocator.Allocate("new_V");
        gpu_sim.Concat(V, values[j], new_V, 0, kInGpuHbm);
        gpu_sim.ReleaseMatrix(V);
        V = new_V;
      }
    }

    // Step 2: Move Q, K, V to SRAM for computation
    gpu_sim.MoveMatrixToSharedMem(current_query);
    gpu_sim.MoveMatrixToSharedMem(K);
    gpu_sim.MoveMatrixToSharedMem(V);

    // Step 3: Transpose K to get K^T
    gpu_sim.Transpose(K, kInSharedMemory);

    // Step 4: Compute Q * K^T (shape: [i+1, i+1])
    Matrix* QK = matrix_memory_allocator.Allocate("QK");
    gpu_sim.MatMul(current_query, K, QK);

    // Step 5: Apply Softmax row-wise to QK
    // For each row, compute softmax
    Matrix* QK_exp = matrix_memory_allocator.Allocate("QK_exp");
    gpu_sim.MatExp(QK, QK_exp);

    // For each row, we need to:
    // 1. Get the row
    // 2. Sum its elements
    // 3. Divide the row by the sum
    Matrix* softmax_QK = nullptr;
    for (size_t row = 0; row < i + 1; ++row) {
      Matrix* row_vec = matrix_memory_allocator.Allocate("row_vec");
      gpu_sim.GetRow(QK_exp, row, row_vec, kInSharedMemory);

      Matrix* row_sum = matrix_memory_allocator.Allocate("row_sum");
      gpu_sim.Sum(row_vec, row_sum);

      Matrix* softmax_row = matrix_memory_allocator.Allocate("softmax_row");
      gpu_sim.MatDiv(row_vec, row_sum, softmax_row);

      if (row == 0) {
        softmax_QK = matrix_memory_allocator.Allocate("softmax_QK");
        gpu_sim.Copy(softmax_row, softmax_QK, kInSharedMemory);
      } else {
        Matrix* new_softmax_QK = matrix_memory_allocator.Allocate("new_softmax_QK");
        gpu_sim.Concat(softmax_QK, softmax_row, new_softmax_QK, 0, kInSharedMemory);
        gpu_sim.ReleaseMatrix(softmax_QK);
        softmax_QK = new_softmax_QK;
      }

      gpu_sim.ReleaseMatrix(row_vec);
      gpu_sim.ReleaseMatrix(row_sum);
      gpu_sim.ReleaseMatrix(softmax_row);
    }

    // Step 6: Transpose V back (we need V in its original form)
    gpu_sim.Transpose(K, kInSharedMemory); // Undo K transpose for cleanup

    // Step 7: Compute softmax_QK * V (shape: [i+1, 512])
    Matrix* result = matrix_memory_allocator.Allocate("result");
    gpu_sim.MatMul(softmax_QK, V, result);

    // Step 8: Move result to HBM
    gpu_sim.MoveMatrixToGpuHbm(result);

    // Clean up
    gpu_sim.ReleaseMatrix(K);
    gpu_sim.ReleaseMatrix(V);
    gpu_sim.ReleaseMatrix(QK);
    gpu_sim.ReleaseMatrix(QK_exp);
    gpu_sim.ReleaseMatrix(softmax_QK);

    gpu_sim.Run(false, &matrix_memory_allocator);
    rater.CommitAnswer(*result);
    /*********************  End of your code *********************/
  
    /*
     * If you want to print debug information, you can use:
     * gpu_sim.Run(true, &matrix_memory_allocator);
     * At the end of your calculation, you should commit the answer:
     * rater.CommitAnswer(YOUR_ANSWER_MATRIX) in each iteration.
     * Your answer matrix should be in GPU HBM.
     * After the answer is committed, the answer matrix will be released
     * automatically.
     */
  }
}

void Test(Rater &rater, GpuSimulator &gpu_sim,
          MatrixMemoryAllocator &matrix_memory_allocator) {
  Calculate(rater.keys_, rater.values_, rater, gpu_sim,
            matrix_memory_allocator);
  rater.PrintResult(gpu_sim);
}

} // namespace sjtu