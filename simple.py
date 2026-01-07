import csv
import itertools
import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm

MODEL_ID = "state-spaces/mamba-130m-hf"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Utils ----------------

def apply_low_rank_(w, rank):
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    w.copy_((U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank]).to(w.dtype))


def fake_quant_(w, bits, group_size):
    qmax = 2 ** (bits - 1) - 1
    wv = w.view(-1, group_size)
    scale = wv.abs().amax(dim=-1, keepdim=True) / qmax
    scale.clamp_(min=1e-8)
    w.copy_(((wv / scale).round().clamp(-qmax, qmax) * scale).view_as(w))


def layer_indices(mode, n_layers):
    if mode == "all":
        return set(range(n_layers))
    if mode == "early":
        return set(range(2, n_layers // 3))
    if mode == "mid":
        return set(range(n_layers // 3, 2 * n_layers // 3))
    if mode == "late":
        return set(range(2 * n_layers // 3, n_layers))
    raise ValueError(mode)


def compress_mamba(model, rb, rc, rdt, bits, gsize, layers):
    for i, layer in enumerate(model.backbone.layers):
        if i not in layers:
            continue

        m = layer.mixer
        DT = m.dt_proj.in_features
        DS = m.A_log.shape[-1]

        W = m.x_proj.weight.data
        _, wb, wc = torch.split(W, [DT, DS, DS], dim=0)

        apply_low_rank_(wb, rb)
        fake_quant_(wb, bits, gsize)

        apply_low_rank_(wc, rc)
        fake_quant_(wc, bits, gsize)

        apply_low_rank_(m.dt_proj.weight.data, rdt)
        fake_quant_(m.dt_proj.weight.data, bits, gsize)


def evaluate(model, ids):
    model.eval()
    nlls = []
    stride, seq_len = 512, 1024

    with torch.inference_mode():
        for i in range(0, min(ids.size(1), 5000), stride):
            end = min(i + seq_len, ids.size(1))
            x = ids[:, end - seq_len : end].to(DEVICE)
            y = x.clone()
            y[:, : -(end - i)] = -100

            logits = model(x).logits
            loss = F.cross_entropy(
                logits[..., :-1, :].reshape(-1, logits.size(-1)),
                y[..., 1:].reshape(-1),
            )
            nlls.append(loss)

    return torch.exp(torch.stack(nlls).mean()).item()

# ---------------- Main ----------------

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    ids = tokenizer(
        "\n\n".join(t for t in dataset["text"] if t.strip()),
        return_tensors="pt",
    ).input_ids

    bits = [8, 4]
    group_sizes = [64, 128, 256]
    ranks_b = [16, 8, 4]
    ranks_c = [16, 8, 4]
    ranks_dt = [48, 36, 24]
    layer_modes = ["early", "mid", "late", "all"]
    # layer_modes = ["early"]

    with open("mamba_full_sweep.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "bits", "group", "rb", "rc", "rdt", "layer_mode", "ppl"
        ])

        for (b, g, rb, rc, rdt, lm) in itertools.product(
            bits, group_sizes, ranks_b, ranks_c, ranks_dt, layer_modes
        ):
            print(f"\n🚀 bits={b}, g={g}, B={rb}, C={rc}, DT={rdt}, layers={lm}")

            model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
            layers = layer_indices(lm, len(model.backbone.layers))

            compress_mamba(model, rb, rc, rdt, b, g, layers)
            ppl = evaluate(model, ids)

            writer.writerow([b, g, rb, rc, rdt, lm, f"{ppl:.4f}"])
            print(f"✅ PPL: {ppl:.4f}")

            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
