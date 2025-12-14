#include <string.h>

// Dimensions
#define M 64
#define N 1024
#define K 4096

void tiled_mul(const float *input,   // [M x N]
               const float *weights, // [N x K]
               float *output         // [M x K]
) {
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem0 depth = \
    65536
#pragma HLS INTERFACE m_axi port = weights offset = slave bundle =             \
    gmem1 depth = 4194304
#pragma HLS INTERFACE m_axi port = output offset = slave bundle =              \
    gmem0 depth = 262144
#pragma HLS INTERFACE s_axilite port = return

  float buff_A[M][N];
#pragma HLS ARRAY_PARTITION variable = buff_A dim = 1 complete

// Load A into BRAM
LOAD_A:
  for (int i = 0; i < M; i++) {
    memcpy(buff_A[i], (const float *)&input[i * N], N * sizeof(float));
  }

// 2. COMPUTE LOOP
// We compute one OUTPUT COLUMN (j) at a time.
// This allows us to stream 'weights' linearly and write 'output' linearly.
COL_LOOP:
  for (int j = 0; j < K; j++) {

    // Accumulator for one column of output (across all batches)
    float acc[M];
#pragma HLS ARRAY_PARTITION variable = acc complete

    for (int i = 0; i < M; i++)
#pragma HLS UNROLL
      acc[i] = 0;

  ROW_LOOP:
    for (int k = 0; k < N; k++) {
#pragma HLS PIPELINE II = 1

      float w_val = weights[k * K + j];

    // Multiply against all cached inputs
    BATCH_CALC:
      for (int i = 0; i < M; i++) {
#pragma HLS UNROLL
        acc[i] += buff_A[i][k] * w_val;
      }
    }

  // Write result column to memory
  WRITE_LOOP:
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE
      output[i * K + j] = acc[i];
    }
  }
}
