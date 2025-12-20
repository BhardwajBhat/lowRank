import csv
import itertools
import time
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

# --- Helpers ---
def rms_norm(x, weight, eps=1e-5):
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * weight

def simulate_quant(w, bits=4, group_size=64):
    orig_dtype, n = w.dtype, w.shape[-1]
    next_pow2 = 2 ** ((n - 1).bit_length())
    w_padded = F.pad(w.float(), (0, next_pow2 - n))
    
    # Build Hadamard
    H = torch.tensor([[1.0]], device=w.device)
    while H.shape[0] < next_pow2:
        H = torch.cat((torch.cat((H, H), 1), torch.cat((H, -H), 1)), 0)
    H = H / torch.sqrt(torch.tensor(next_pow2, device=w.device))
    
    w_rot = w_padded @ H
    
    # Group quantization
    group_size = min(group_size, n)
    pad_size = (group_size - (w_rot.numel() % group_size)) % group_size
    w_flat = F.pad(w_rot.reshape(-1), (0, pad_size)).reshape(-1, group_size)
    
    q_max = 2 ** (bits - 1) - 1
    scale = w_flat.abs().max(-1, keepdim=True)[0] / q_max
    scale = scale.clamp(min=1e-8)
    w_quant = (w_flat / scale).round().clamp(-q_max, q_max) * scale
    
    w_rot = w_quant.reshape(-1)[:w_rot.numel()].reshape(w_rot.shape)
    w_out = (w_rot @ H)[..., :n]
    return w_out.to(orig_dtype)

def apply_low_rank(w, rank):
    U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
    return (U[:, :rank] @ torch.diag(S[:rank]) @ Vh[:rank, :]).to(w.dtype)

# --- Model ---
class ProjLayer(nn.Module):
    def __init__(self, weight, bias=None, config=None):
        super().__init__()
        config = config or {}
        w = weight.data.clone()
        
        if config.get("svd_rank"):
            w = apply_low_rank(w, config["svd_rank"])
        if config.get("quant"):
            w = simulate_quant(w, config.get("bits", 4), config.get("group_size", 64))
        
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(bias.data.clone()) if bias is not None else None
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class UnifiedMambaBlock(nn.Module):
    def __init__(self, hf_layer, config):
        super().__init__()
        m = hf_layer.mixer
        w_dt, w_b, w_c = torch.split(m.x_proj.weight.data, [DT_RANK, D_STATE, D_STATE], 0)
        
        self.in_proj = ProjLayer(m.in_proj.weight, m.in_proj.bias, config.get("in_proj"))
        self.out_proj = ProjLayer(m.out_proj.weight, m.out_proj.bias, config.get("out_proj"))
        self.dt_proj = ProjLayer(m.dt_proj.weight, m.dt_proj.bias, config.get("dt_proj"))
        self.x_proj_dt = ProjLayer(w_dt, None, config.get("x_proj_dt"))
        self.x_proj_b = ProjLayer(w_b, None, config.get("x_proj_b"))
        self.x_proj_c = ProjLayer(w_c, None, config.get("x_proj_c"))
        
        self.conv1d = nn.Conv1d(D_INNER, D_INNER, D_CONV, groups=D_INNER, padding=D_CONV - 1)
        self.conv1d.weight.data.copy_(m.conv1d.weight.data)
        self.conv1d.bias.data.copy_(m.conv1d.bias.data)
        
        self.A_log, self.D = nn.Parameter(m.A_log.data), nn.Parameter(m.D.data)
    
    def forward(self, u):
        x, z = self.in_proj(u).chunk(2, -1)
        x = F.silu(self.conv1d(x.transpose(1, 2))[:, :, :u.size(1)].transpose(1, 2))
        
        dt = F.softplus(self.dt_proj(self.x_proj_dt(x)))
        B, C = self.x_proj_b(x), self.x_proj_c(x)
        A = -torch.exp(self.A_log.float())
        
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)
        
        h = torch.zeros(u.size(0), D_INNER, D_STATE, device=u.device)
        y_list = []
        for t in range(u.size(1)):
            h = h * dA[:, t] + x[:, t].unsqueeze(-1) * dB[:, t]
            y_list.append((h @ C[:, t].unsqueeze(-1)).squeeze(-1) + (self.D * x[:, t]))
        
        return self.out_proj(torch.stack(y_list, 1) * F.silu(z))

class MambaModelUnified(nn.Module):
    def __init__(self, hf_model, config, skip_layers=[]):
        super().__init__()
        self.embedding = hf_model.backbone.embeddings
        self.final_norm = hf_model.backbone.norm_f
        self.lm_head = hf_model.lm_head
        self.layers = nn.ModuleList([
            UnifiedMambaBlock(l, config if i not in skip_layers else {})
            for i, l in enumerate(hf_model.backbone.layers)
        ])
        self.norms = nn.ModuleList([l.norm for l in hf_model.backbone.layers])
    
    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer, norm in zip(self.layers, self.norms):
            x = layer(rms_norm(x, norm.weight)) + x
        return self.lm_head(rms_norm(x, self.final_norm.weight))

# --- Evaluation ---
def evaluate(model, encodings, name, max_tokens=5000):
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    start = time.time()
    
    nlls, stride, seq_len = [], 512, 1024
    with torch.inference_mode():
        for i in tqdm.tqdm(range(0, min(encodings.size(1), max_tokens), stride), desc=name, leave=False):
            end = min(i + seq_len, encodings.size(1))
            inp = encodings[:, end - seq_len:end].to(DEVICE)
            tgt = inp.clone()
            tgt[:, :-(end - i)] = -100
            
            out = model(inp)
            logits = out.logits if hasattr(out, "logits") else out
            loss = F.cross_entropy(logits[..., :-1, :].reshape(-1, logits.size(-1)), tgt[..., 1:].reshape(-1))
            nlls.append(loss)
    
    ppl = torch.exp(torch.stack(nlls).mean()).item()
    elapsed = time.time() - start
    mem = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
    
    return ppl, elapsed, mem

# --- Main ---
def main():
    print("🔧 Generating SVD grid search experiments...")
    
    # Define search space
    b_ranks = [12, 8, 4]
    c_ranks = [12, 8, 4]
    dt_ranks = [48, 36, 24]
    skip_layers = [0, 1, 22, 23]
    
    # Generate all combinations
    experiments = [{"name": "Baseline", "config": {}, "skip": []}]
    
    for b, c, dt in itertools.product(b_ranks, c_ranks, dt_ranks):
        exp = {
            "name": f"B{b}_C{c}_DT{dt}",
            "config": {
                "x_proj_b": {"svd_rank": b},
                "x_proj_c": {"svd_rank": c},
                "dt_proj": {"svd_rank": dt}
            },
            "skip": skip_layers
        }
        experiments.append(exp)
    
    print(f"📊 Total experiments: {len(experiments)} (1 baseline + {len(experiments)-1} SVD configs)")
    print(f"   B ranks: {b_ranks}")
    print(f"   C ranks: {c_ranks}")
    print(f"   DT ranks: {dt_ranks}")
    print(f"   Skipping layers: {skip_layers}\n")
    
    # Load model and data
    print("📥 Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()
    
    print("📥 Loading dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(t for t in dataset["text"] if t.strip()), return_tensors="pt").input_ids
    
    print(f"✅ Setup complete. Device: {DEVICE}\n")
    print("=" * 70)
    
    # Run experiments
    with open("svd_grid_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "B_Rank", "C_Rank", "DT_Rank", "Perplexity", "Time(s)", "Memory(GB)"])
        
        for idx, exp in enumerate(experiments):
            print(f"\n[{idx+1}/{len(experiments)}] 🚀 Running: {exp['name']}")
            
            try:
                if exp['name'] == "Baseline":
                    model = hf_model
                    b_val, c_val, dt_val = "Full", "Full", "Full"
                else:
                    model = MambaModelUnified(hf_model, exp['config'], exp['skip']).to(DEVICE)
                    b_val = exp['config']['x_proj_b']['svd_rank']
                    c_val = exp['config']['x_proj_c']['svd_rank']
                    dt_val = exp['config']['dt_proj']['svd_rank']
                
                ppl, elapsed, mem = evaluate(model, encodings, exp['name'])
                writer.writerow([exp['name'], b_val, c_val, dt_val, f"{ppl:.4f}", f"{elapsed:.2f}", f"{mem:.3f}"])
                f.flush()  # Save immediately
                
                print(f"   ✅ PPL: {ppl:.4f} | Time: {elapsed:.1f}s | Mem: {mem:.2f}GB")
                
                if exp['name'] != "Baseline":
                    del model
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
            except Exception as e:
                print(f"   ❌ Error: {e}")
                writer.writerow([exp['name'], "ERROR", "ERROR", "ERROR", "ERROR", "ERROR", "ERROR"])
                f.flush()
    
    print("\n" + "=" * 70)
    print("🎉 Grid search complete! Results saved to: svd_grid_results.csv")
    print("\n📊 Quick summary:")
    
    # Print a quick summary
    import pandas as pd
    try:
        df = pd.read_csv("svd_grid_results.csv")
        df['Perplexity'] = pd.to_numeric(df['Perplexity'], errors='coerce')
        
        print(f"\nBest 5 configurations by perplexity:")
        print(df.nsmallest(5, 'Perplexity')[['Name', 'Perplexity', 'Memory(GB)']].to_string(index=False))
        
        print(f"\nLowest 5 memory usage:")
        df_valid = df[df['Memory(GB)'] != 'ERROR']
        df_valid['Memory(GB)'] = pd.to_numeric(df_valid['Memory(GB)'], errors='coerce')
        print(df_valid.nsmallest(5, 'Memory(GB)')[['Name', 'Perplexity', 'Memory(GB)']].to_string(index=False))
        
    except:
        print("(Install pandas for automatic summary: pip install pandas)")

if __name__ == "__main__":
    main()