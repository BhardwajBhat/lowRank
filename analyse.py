import pandas as pd
from matplotlib import pyplot as plt

# Load CSV
df = pd.read_csv("./mamba_full_sweep.csv")

# Ensure numeric
df["ppl"] = df["ppl"].astype(float)

df = df[~df["layer_mode"].isin(["late", "all"])]

print("\n=== BASIC STATS ===")
print(df.describe())

# ------------------------------------------------
# Best overall configs
# ------------------------------------------------
print("\n=== TOP 10 BEST (LOWEST PPL) ===")
print(df.sort_values("ppl").head(10).to_string(index=False))


# ------------------------------------------------
# Average PPL per dimension
# ------------------------------------------------
def avg_by(col):
    return df.groupby(col)["ppl"].mean().sort_values()


print("\n=== AVG PPL BY BITS ===")
print(avg_by("bits"))

print("\n=== AVG PPL BY GROUP SIZE ===")
print(avg_by("group"))

print("\n=== AVG PPL BY RB ===")
print(avg_by("rb"))

print("\n=== AVG PPL BY RC ===")
print(avg_by("rc"))

print("\n=== AVG PPL BY RDT ===")
print(avg_by("rdt"))

print("\n=== AVG PPL BY LAYER MODE ===")
print(avg_by("layer_mode"))

# ------------------------------------------------
# Sensitivity: how much PPL changes
# ------------------------------------------------
print("\n=== PPL RANGE BY FACTOR ===")
for col in ["bits", "group", "rb", "rc", "rdt", "layer_mode"]:
    spread = df.groupby(col)["ppl"].mean().max() - df.groupby(col)["ppl"].mean().min()
    print(f"{col:12s}: ΔPPL = {spread:.4f}")

# ------------------------------------------------
# Near-baseline configs (within +0.1 PPL of best)
# ------------------------------------------------
best = df["ppl"].min()
print(f"\n=== NEAR-OPTIMAL (PPL <= {best + 0.1:.3f}) ===")
print(df[df["ppl"] <= best + 0.1].sort_values("ppl").to_string(index=False))


plt.figure()
for col, label in [("rb", "B"), ("rc", "C"), ("rdt", "DT")]:
    means = df.groupby(col)["ppl"].mean()
    plt.plot(means.index, means.values, marker="o", label=label)

plt.xlabel("SVD rank")
plt.ylabel("Perplexity")
plt.title("SVD Rank vs PPL (average)")
plt.legend()
plt.grid(True)
plt.show()

safe = df[(df["rb"] == 16) & (df["rc"] == 16) & (df["bits"] == 8) & (df["group"] == 64)]

print(safe.groupby("rdt")["ppl"].mean())


# Filter for the specific condition: rank of b (rb) and c (rc) at 16
# We also focus on 4-bit and 8-bit quantization
filtered_df = df[(df["rb"] == 16) & (df["rc"] == 16) & (df["bits"].isin([4, 8]))]

# Group by rdt and bits to get the mean PPL for each combination
# This creates a table where rows are rdt and columns are bits (4, 8)
plot_data = filtered_df.groupby(["rdt", "bits"])["ppl"].mean().unstack()

# Plotting
plt.figure(figsize=(10, 6))

if 4 in plot_data.columns:
    plt.plot(
        plot_data.index, plot_data[4], marker="o", linestyle="-", label="Quant 4-bit"
    )
if 8 in plot_data.columns:
    plt.plot(
        plot_data.index, plot_data[8], marker="s", linestyle="--", label="Quant 8-bit"
    )

plt.xlabel("Rank of DT (rdt)")
plt.ylabel("Perplexity (PPL)")
plt.title("PPL vs DT Rank for 4-bit and 8-bit Quantization (rb=16, rc=16)")
plt.legend(title="Quantization")
plt.grid(True, linestyle="--", alpha=0.7)
plt.tight_layout()

# Save the plot
plt.savefig("ppl_vs_rdt_quantization.png")
plt.show()

# Optional: Print the values to verify
print("=== Mean PPL by RDT and Bits (rb=16, rc=16) ===")
print(plot_data)
