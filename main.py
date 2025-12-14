import numpy as np
import time

# --- Configuration ---
M, N, K = 64, 1024, 4090  # Input: [64, 1024], Weights: [1024, 4090]
TARGET_RANK = 256  # The rank we want to compress to

print(f"Dimensions: Input[{M}x{N}] * Weights[{N}x{K}]")
print(f"Comparing Standard Tiled Mul vs. SVD (Rank {TARGET_RANK})...")
print("-" * 60)

# --- 1. Data Generation (Make W suitable for Rank 256) ---
# To make SVD a "fair" comparison, W must actually BE low rank (or close to it).
# We construct W by multiplying two smaller matrices: [1024x256] @ [256x4090]
np.random.seed(42)
U_gen = np.random.randn(N, TARGET_RANK).astype(np.float32)
V_gen = np.random.randn(TARGET_RANK, K).astype(np.float32)
W_fixed = np.dot(U_gen, V_gen)  # This is our "Fat" Fixed Matrix (Rank 256)

# The input batch (Input activations)
A_input = np.random.randn(M, N).astype(np.float32)


# --- 2. Offline Preparation (The "Setup" Cost) ---
# We do this ONCE before deployment. This does not count towards inference latency.
print("Performing Offline SVD Decomposition...", end="")
t0 = time.time()

# Compute SVD: W approx U * S * Vt
# full_matrices=False makes it faster (Economy SVD)
U, S, Vt = np.linalg.svd(W_fixed, full_matrices=False)

# Truncate to keep only top 256 singular values
U_k = U[:, :TARGET_RANK]  # Shape: [1024, 256]
S_k = np.diag(S[:TARGET_RANK])  # Shape: [256, 256]
Vt_k = Vt[:TARGET_RANK, :]  # Shape: [256, 4090]

# OPTIMIZATION: Merge S into Vt so we have fewer multiplications at runtime
# We define Left = U_k, Right = (S_k @ Vt_k)
W_decomposed_L = U_k  # [1024, 256]
W_decomposed_R = np.dot(S_k, Vt_k)  # [256, 4090]

print(f" Done ({time.time() - t0:.4f}s)")
print("-" * 60)


# --- 3. Benchmarking Inference Speed ---
iterations = 1000  # Run many times to get average speed

res_standard = np.zeros((M, K), dtype=np.float32)
res_svd = np.zeros((M, K), dtype=np.float32)

# Method A: Standard Tiled Matmul (Numpy uses BLAS tiling under the hood)
start_time = time.time()
for _ in range(iterations):
    res_standard = np.dot(A_input, W_fixed)
std_time = (time.time() - start_time) / iterations

# Method B: SVD Low-Rank Multiplication
# Logic: (Input @ L) @ R -> The parentheses are crucial for speed!
# Shape change: [64,1024] @ [1024,256] -> [64,256] @ [256,4090] -> [64,4090]
start_time = time.time()
for _ in range(iterations):
    # We explicitly enforce order of operations for max speed
    intermediate = np.dot(A_input, W_decomposed_L)
    res_svd = np.dot(intermediate, W_decomposed_R)
svd_time = (time.time() - start_time) / iterations


# --- 4. Results Analysis ---

# Accuracy Check (Frobenius Norm of the difference)
diff = np.linalg.norm(res_standard - res_svd)
print("Results Analysis:")
print(f"1. Standard Tiled Time : {std_time * 1000:.4f} ms")
print(f"2. SVD (Rank {TARGET_RANK}) Time  : {svd_time * 1000:.4f} ms")
print(f"   -> Speedup          : {std_time / svd_time:.2f}x FASTER")
print(f"3. Approximation Error : {diff:.6f} (Lower is better)")

if diff < 1e-3:
    print("\n[SUCCESS] The SVD approximation is extremely accurate.")
else:
    print(
        "\n[NOTE] Some accuracy loss detected (expected if W was not perfectly low rank)."
    )
