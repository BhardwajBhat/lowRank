import csv
import os
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration ---
MODEL_ID = "state-spaces/mamba-130m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

D_MODEL, EXPAND = 768, 2
D_INNER = D_MODEL * EXPAND
D_STATE, D_CONV = 16, 4
DT_RANK = D_MODEL // 16

torch.manual_seed(42)

# --- Clean Helpers ---


def rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight


def simulate_quant(w: torch.Tensor, bits=4, group_size=64):
    """
    Hadamard rotation followed by Group-wise Fake Quantization.
    group_size: Number of elements sharing a single scale factor.
    """
    orig_dtype = w.dtype
    n = w.shape[-1]

    # 1. Hadamard requires power-of-2 size
    next_pow2 = 2 ** ((n - 1).bit_length())
    w_padded = F.pad(w.float(), (0, next_pow2 - n))

    # 2. Iteratively build the Hadamard Matrix
    H = torch.tensor([[1.0]], device=w.device)
    while H.shape[0] < next_pow2:
        H = torch.cat((torch.cat((H, H), dim=1), torch.cat((H, -H), dim=1)), dim=0)
    H = H / torch.sqrt(torch.tensor(next_pow2, device=w.device))

    # 3. Transform to Hadamard domain
    w_rotated = w_padded @ H

    # --- 4. Group-wise Quantization Logic ---
    # Reshape to [Total_Elements / Group_Size, Group_Size]
    # This treats every 'group_size' chunk as its own quantization unit
    orig_rotated_shape = w_rotated.shape
    w_flat = w_rotated.reshape(-1, group_size)

    q_max = 2 ** (bits - 1) - 1

    # Calculate scale per group
    # scale shape: [Total_Elements / Group_Size, 1]
    scale = w_flat.abs().max(dim=-1, keepdim=True)[0] / q_max
    scale = scale.clamp(min=1e-8)

    # Quantize and Dequantize
    w_quant = (w_flat / scale).round().clamp(-q_max, q_max) * scale

    # Restore shape
    w_rotated = w_quant.reshape(orig_rotated_shape)
    # ----------------------------------------

    # 5. Transform back and crop
    w_out = (w_rotated @ H)[..., :n]

    return w_out.to(orig_dtype)


def apply_low_rank(w: torch.Tensor, rank: int):
    """Apply SVD approximation."""
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    return (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).to(w.dtype)


# --- Modular Core ---


class ProjLayer(nn.Module):
    def __init__(self, weight, bias=None, config=None):
        super().__init__()
        config = config or {}
        w = weight.data.clone()

        # 1. Apply SVD if requested
        if config.get("svd_rank"):
            w = apply_low_rank(w, config["svd_rank"])

        # 2. Apply Simulated Quantization if requested
        if config.get("quant"):
            w = simulate_quant(w)

        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(bias.data.clone()) if bias is not None else None

    def forward(self, x):
        return F.linear(x, self.weight, self.bias)


class UnifiedMambaBlock(nn.Module):
    def __init__(self, hf_layer, config):
        super().__init__()
        m = hf_layer.mixer
        # Split x_proj weight
        w_dt, w_b, w_c = torch.split(
            m.x_proj.weight.data, [DT_RANK, D_STATE, D_STATE], dim=0
        )

        self.in_proj = ProjLayer(
            m.in_proj.weight, m.in_proj.bias, config.get("in_proj")
        )
        self.out_proj = ProjLayer(
            m.out_proj.weight, m.out_proj.bias, config.get("out_proj")
        )
        self.dt_proj = ProjLayer(
            m.dt_proj.weight, m.dt_proj.bias, config.get("dt_proj")
        )
        self.x_proj_dt = ProjLayer(w_dt, None, config.get("x_proj_dt"))
        self.x_proj_b = ProjLayer(w_b, None, config.get("x_proj_b"))
        self.x_proj_c = ProjLayer(w_c, None, config.get("x_proj_c"))

        self.conv1d = nn.Conv1d(
            D_INNER, D_INNER, D_CONV, groups=D_INNER, padding=D_CONV - 1
        )
        self.conv1d.weight.data.copy_(m.conv1d.weight.data)
        self.conv1d.bias.data.copy_(m.conv1d.bias.data)
        self.A_log, self.D = nn.Parameter(m.A_log.data), nn.Parameter(m.D.data)

    def forward(self, u):
        x, z = self.in_proj(u).chunk(2, dim=-1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, : u.size(1)].transpose(1, 2))

        dt = F.softplus(self.dt_proj(self.x_proj_dt(x)))
        B, C, A = self.x_proj_b(x), self.x_proj_c(x), -torch.exp(self.A_log.float())

        dA, dB = torch.exp(dt.unsqueeze(-1) * A), dt.unsqueeze(-1) * B.unsqueeze(-2)

        h = torch.zeros(u.size(0), D_INNER, D_STATE, device=u.device)
        y_list = []
        for t in range(u.size(1)):
            h = h * dA[:, t] + x[:, t].unsqueeze(-1) * dB[:, t]
            y_list.append((h @ C[:, t].unsqueeze(-1)).squeeze(-1) + (self.D * x[:, t]))

        return self.out_proj(torch.stack(y_list, dim=1) * F.silu(z))


class MambaModelUnified(nn.Module):
    def __init__(self, hf_model, config, skip_layers):
        super().__init__()
        self.embedding = hf_model.backbone.embeddings
        self.final_norm = hf_model.backbone.norm_f
        self.lm_head = hf_model.lm_head
        self.layers = nn.ModuleList(
            [
                UnifiedMambaBlock(l, config if i not in skip_layers else {})
                for i, l in enumerate(hf_model.backbone.layers)
            ]
        )
        self.norms = nn.ModuleList([l.norm for l in hf_model.backbone.layers])

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer, norm in zip(self.layers, self.norms):
            x = layer(rms_norm(x, norm.weight)) + x
        return self.lm_head(rms_norm(x, self.final_norm.weight))


# --- Evaluation ---


import csv
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Reuse your existing MODEL_ID, DEVICE, and Helper Classes ---
# (Ensure ProjLayer, UnifiedMambaBlock, MambaModelUnified are defined above)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

    csv_file = "mamba_combinatorial_svd.csv"

    # Define the rank search space
    ranks_b = [16, 8, 4]
    ranks_c = [16, 8, 4]
    ranks_dt = [48, 36, 24]

    # Generate all combinations
    combinations = list(itertools.product(ranks_b, ranks_c, ranks_dt))

    # Prepare Data
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer(
        "\n\n".join(t for t in dataset["text"] if t.strip()), return_tensors="pt"
    ).input_ids

    def evaluate(model, name):
        nlls, stride, seq_len = [], 512, 1024
        with torch.inference_mode():
            # Evaluation limited to 5000 tokens for experiment speed
            for i in tqdm.tqdm(
                range(0, min(encodings.size(1), 5000), stride),
                desc=f"Eval {name}",
                leave=False,
            ):
                end_loc = min(i + seq_len, encodings.size(1))
                input_ids = encodings[:, end_loc - seq_len : end_loc].to(DEVICE)
                target_ids = input_ids.clone()
                target_ids[:, : -(end_loc - i)] = -100
                out = model(input_ids)
                logits = out.logits if hasattr(out, "logits") else out
                loss = F.cross_entropy(
                    logits[..., :-1, :].reshape(-1, logits.size(-1)),
                    target_ids[..., 1:].reshape(-1),
                )
                nlls.append(loss)
        return torch.exp(torch.stack(nlls).mean()).item()

    # Execution loop
    with open(csv_file, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["B_Rank", "C_Rank", "DT_Rank", "Perplexity"])

        # 1. Run Baseline
        print("🚀 Running Baseline...")
        base_ppl = evaluate(hf_model, "Baseline")
        writer.writerow(["Full", "Full", "Full", f"{base_ppl:.4f}"])

        # 2. Run All Combinations
        for rb, rc, rdt in combinations:
            run_name = f"B{rb}_C{rc}_DT{rdt}"
            print(f"🚀 Running: {run_name}")

            config = {
                "x_proj_b": {"svd_rank": rb, "quant": 4},
                "x_proj_c": {"svd_rank": rc, "quant": 4},
                "dt_proj": {"svd_rank": rdt, "quant": 4},
            }

            # Apply SVD to custom model (skipping sensitive layers)
            custom_model = MambaModelUnified(
                hf_model, config, skip_layers=[0, 1, 22, 23]
            ).to(DEVICE)
            ppl = evaluate(custom_model, run_name)

            writer.writerow([rb, rc, rdt, f"{ppl:.4f}"])
            print(f"✅ {run_name} -> PPL: {ppl:.4f}")

            del custom_model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
