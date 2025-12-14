#include <string.h>

#define M 64
#define N 1024
#define R 256 // Rank
#define K 4096

void svd_mul(const float *input,
             const float *wl, // [N x R]
             const float *wr, // [R x K]
             float *output) {
#pragma HLS INTERFACE m_axi port = input offset = slave bundle = gmem0 depth = \
    65536
#pragma HLS INTERFACE m_axi port = wl offset = slave bundle = gmem1 depth =    \
    262144
#pragma HLS INTERFACE m_axi port = wr offset = slave bundle = gmem0 depth =    \
    1048576
#pragma HLS INTERFACE m_axi port = output offset = slave bundle =              \
    gmem1 depth = 262144
#pragma HLS INTERFACE s_axilite port = return

  // BUFFERS
  float buff_A[M][N];
#pragma HLS ARRAY_PARTITION variable = buff_A dim = 1 complete

  float buff_Inter[M][R];
#pragma HLS ARRAY_PARTITION variable = buff_Inter dim = 1 complete

LOAD_A:
  for (int i = 0; i < M; i++) {
    memcpy(buff_A[i], (const float *)&input[i * N], N * sizeof(float));
  }

  // 2. STAGE 1 (A * WL -> Inter)
  for (int i = 0; i < M; i++) {
    for (int r = 0; r < R; r++) {
#pragma HLS PIPELINE
      buff_Inter[i][r] = 0;
    }
  }

// Loop over Inputs (N) then Rank (R) to stream WL linearly
S1_N_LOOP:
  for (int n = 0; n < N; n++) {

    // Cache column of input for all batches
    float in_col[M];
#pragma HLS ARRAY_PARTITION variable = in_col complete
    for (int i = 0; i < M; i++)
#pragma HLS UNROLL
      in_col[i] = buff_A[i][n];

  S1_R_LOOP:
    for (int r = 0; r < R; r++) {
#pragma HLS PIPELINE II = 1
      float w_val = wl[n * R + r];

      // Update all batches
      for (int i = 0; i < M; i++) {
#pragma HLS UNROLL
        buff_Inter[i][r] += in_col[i] * w_val;
      }
    }
  }

// 3. STAGE 2 (Inter * WR -> Output)
// We compute one output column (k) at a time
S2_K_LOOP:
  for (int k = 0; k < K; k++) {

    float acc[M];
#pragma HLS ARRAY_PARTITION variable = acc complete
    for (int i = 0; i < M; i++)
#pragma HLS UNROLL
      acc[i] = 0;

  S2_R_LOOP:
    for (int r = 0; r < R; r++) {
#pragma HLS PIPELINE II = 1
      float w_val = wr[r * K + k];

      // Multiply against intermediate buffer
      for (int i = 0; i < M; i++) {
#pragma HLS UNROLL
        acc[i] += buff_Inter[i][r] * w_val;
      }
    }

    // Write Output
    for (int i = 0; i < M; i++) {
#pragma HLS PIPELINE
      output[i * K + k] = acc[i];
    }
  }
}
