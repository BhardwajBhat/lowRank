import csv
import itertools
import os
import time
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "state-spaces/mamba-130m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Utils ----------------
def fwht(x):
    """
    In-place Fast Walsh–Hadamard Transform
    x: (..., n), n must be power of 2
    """
    n = x.shape[-1]
    h = 1
    while h < n:
        a = x[..., ::2*h]
        b = x[..., h::2*h]
        x[..., ::2*h] = a + b
        x[..., h::2*h] = a - b
        h *= 2
    return x

def apply_low_rank_(w, rank):
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    w.copy_((U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank]).to(w.dtype))

def fake_quant_(w, bits, group_size):
    qmax = 2 ** (bits - 1) - 1
    wv = w.view(-1, group_size)
    scale = wv.abs().amax(dim=-1, keepdim=True) / qmax
    scale.clamp_(min=1e-8)
    w.copy_(((wv / scale).round().clamp(-qmax, qmax) * scale).view_as(w))

def hadamard_quant_(w: torch.Tensor, bits=4, group_size=64):
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

# def hadamard_quant_(w, bits, group):
#     """
#     In-place Hadamard + group quant
#     SAFE: no explicit Hadamard matrix
#     """
#     device = w.device
#     orig_dtype = w.dtype

#     # Move to CPU (very important)
#     x = w.float().cpu()

#     n = x.shape[-1]
#     p2 = 1 << (n - 1).bit_length()
#     if p2 != n:
#         x = F.pad(x, (0, p2 - n))

#     # Hadamard
#     fwht(x)
#     x /= p2 ** 0.5

#     # Group quant
#     qmax = 2 ** (bits - 1) - 1
#     xg = x.view(-1, group)
#     scale = xg.abs().max(dim=1, keepdim=True)[0].clamp(min=1e-8) / qmax
#     xg = (xg / scale).round().clamp(-qmax, qmax) * scale
#     x = xg.view(x.shape)

#     # Inverse Hadamard
#     fwht(x)
#     x /= p2 ** 0.5

#     # Crop and restore
#     w.copy_(x[..., :n].to(device=device, dtype=orig_dtype))


def layer_indices(mode, n_layers):
    if mode == "all": return set(range(n_layers))
    if mode == "early": return set(range(2, n_layers // 3))
    if mode == "mid": return set(range(n_layers // 3, 2*n_layers // 3))
    if mode == "late": return set(range(2*n_layers // 3, n_layers))
    raise ValueError(mode)

def compress_mamba(model, rb, rc, rdt, bits, gsize, layers, quant_type):
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers: continue
        m = layer.mixer
        DT = m.dt_proj.in_features
        DS = m.A_log.shape[-1]
        W = m.x_proj.weight.data
        _, wb, wc = torch.split(W, [DT, DS, DS], dim=0)

        # Apply SVD
        apply_low_rank_(wb, rb)
        apply_low_rank_(wc, rc)
        apply_low_rank_(m.dt_proj.weight.data, rdt)

        # Apply quant
        if quant_type == "fake":
            fake_quant_(wb, bits, gsize)
            fake_quant_(wc, bits, gsize)
            fake_quant_(m.dt_proj.weight.data, bits, gsize)
        elif quant_type == "hadamard":
            hadamard_quant_(wb, bits, gsize)
            hadamard_quant_(wc, bits, gsize)
            hadamard_quant_(m.dt_proj.weight.data, bits, gsize)
        else:
            raise ValueError(quant_type)

def evaluate(model, ids):
    model.eval()
    nlls = []
    stride, seq_len = 512, 1024
    with torch.inference_mode():
        for i in range(0, min(ids.size(1), 5000), stride):
            end = min(i + seq_len, ids.size(1))
            x = ids[:, end - seq_len:end].to(DEVICE)
            y = x.clone()
            y[:, :-(end - i)] = -100
            logits = model(x).logits
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                y[..., 1:].reshape(-1)
            )
            nlls.append(loss)
    return torch.exp(torch.stack(nlls).mean()).item()

# ---------------- Main ----------------

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer(
        "\n\n".join(t for t in dataset["text"] if t.strip()),
        return_tensors="pt"
    ).input_ids

    bits_list = [8, 4]
    group_sizes = [64, 128]
    ranks_b = [16, 8, 4]
    ranks_c = [16, 8, 4]
    ranks_dt = [48, 36, 24, 12, 6]
    layer_modes = ["early", "mid", "late", "all"]
    quant_types = ["hadamard", "fake"]

    total = len(bits_list) * len(group_sizes) * len(ranks_b) * len(ranks_c) * \
            len(ranks_dt) * len(layer_modes) * len(quant_types)
    done = 0

    csv_file = "hadamard_sweep.csv"
    first_run = not os.path.exists(csv_file)
    with open(csv_file, "a", newline="", buffering=1) as f:
        writer = csv.writer(f)
        if first_run:
            writer.writerow(["quant", "bits", "group", "rb", "rc", "rdt", "layer_mode", "ppl"])
            f.flush()

        for (qt, b, g, rb, rc, rdt, lm) in tqdm(itertools.product(
            quant_types, bits_list, group_sizes, ranks_b, ranks_c, ranks_dt, layer_modes
        ), total=total):
            start_time = time.time()
            model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
            layers = layer_indices(lm, len(model.backbone.layers))
            compress_mamba(model, rb, rc, rdt, b, g, layers, qt)
            ppl = evaluate(model, ids)
            writer.writerow([qt, b, g, rb, rc, rdt, lm, f"{ppl:.4f}"])
            f.flush()
            done += 1
            elapsed = time.time() - start_time
            print(f"✅ Done {done}/{total} | Last run {elapsed:.1f}s | PPL {ppl:.4f}")

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
