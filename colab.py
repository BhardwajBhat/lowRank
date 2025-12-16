import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration & Hyperparameters ---
MODEL_ID = "state-spaces/mamba-130m-hf"

# Mamba-130m Architecture Constants
D_MODEL = 768
EXPAND = 2
D_INNER = D_MODEL * EXPAND  # 1536
D_STATE = 16
D_CONV = 4
DT_RANK = D_MODEL // 16
N_LAYERS = 24

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)

# --- Helper Functions ---

def rms_norm(x, weight, eps=1e-5):
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(mean_sq + eps)
    return normed * weight

# --- The Core Logic: Unified Mamba Block (Updated) ---

class UnifiedMambaBlock(nn.Module):
    def __init__(self, layer_idx, config):
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.params = nn.ParameterDict()

        # Metadata to track operations. Note the new split keys.
        self.layer_meta = {k: {"type": "standard"} for k in [
            "in_proj", "dt_proj", "out_proj", 
            "x_proj_dt", "x_proj_b", "x_proj_c" # <--- Split keys
        ]}

    def _quantize(self, tensor, quantile=0.999):
        t_abs = tensor.abs()
        threshold = torch.quantile(t_abs.float(), quantile)
        scale = threshold / 127.0
        int8_val = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return int8_val, scale

    def load_weights(self, hf_layer):
        mixer = hf_layer.mixer

        # --- 1. Load Static SSM Params ---
        self.params["conv_weight"] = nn.Parameter(mixer.conv1d.weight.data)
        if mixer.conv1d.bias is not None:
            self.params["conv_bias"] = nn.Parameter(mixer.conv1d.bias.data)
        self.params["A_log"] = nn.Parameter(mixer.A_log.data)
        self.params["D"] = nn.Parameter(mixer.D.data)

        # --- 2. Prepare Weights for Processing ---
        # We manually split x_proj here before the loop
        x_proj_w = mixer.x_proj.weight.data
        # x_proj maps to [dt, B, C]
        w_dt, w_b, w_c = torch.split(x_proj_w, [DT_RANK, D_STATE, D_STATE], dim=0)

        # Create a list of (name, weight, bias) to iterate over uniformly
        # Note: x_proj usually has no bias in Mamba, but we handle None safely
        weights_to_process = [
            ("in_proj", mixer.in_proj.weight.data, mixer.in_proj.bias),
            ("dt_proj", mixer.dt_proj.weight.data, mixer.dt_proj.bias),
            ("out_proj", mixer.out_proj.weight.data, mixer.out_proj.bias),
            # The split layers (bias is likely None for these)
            ("x_proj_dt", w_dt, None), 
            ("x_proj_b",  w_b,  None),
            ("x_proj_c",  w_c,  None),
        ]

        # --- 3. Process Projections (SVD / Quant / Standard) ---
        for name, W, bias_data in weights_to_process:
            
            # Get config for this specific projection
            cfg = self.config.get(name, {})
            rank = cfg.get("svd_rank", None)
            do_quant = cfg.get("quant", False)

            # --- SVD ENABLED ---
            if rank is not None:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]

                if do_quant:
                    u_int8, u_scale = self._quantize(U)
                    vh_int8, vh_scale = self._quantize(Vh)
                    self.params[f"{name}_U_int8"] = nn.Parameter(u_int8, requires_grad=False)
                    self.params[f"{name}_U_scale"] = nn.Parameter(u_scale, requires_grad=False)
                    self.params[f"{name}_Vh_int8"] = nn.Parameter(vh_int8, requires_grad=False)
                    self.params[f"{name}_Vh_scale"] = nn.Parameter(vh_scale, requires_grad=False)
                    self.params[f"{name}_S"] = nn.Parameter(S)
                    self.layer_meta[name]["type"] = "svd_quant"
                else:
                    self.params[f"{name}_U"] = nn.Parameter(U)
                    self.params[f"{name}_S"] = nn.Parameter(S)
                    self.params[f"{name}_Vh"] = nn.Parameter(Vh)
                    self.layer_meta[name]["type"] = "svd"

            # --- NO SVD ---
            else:
                if do_quant:
                    w_int8, w_scale = self._quantize(W)
                    self.params[f"{name}_weight_int8"] = nn.Parameter(w_int8, requires_grad=False)
                    self.params[f"{name}_scale"] = nn.Parameter(w_scale, requires_grad=False)
                    self.layer_meta[name]["type"] = "quant"
                else:
                    self.params[f"{name}_weight"] = nn.Parameter(W)
                    self.layer_meta[name]["type"] = "standard"

            if bias_data is not None:
                self.params[f"{name}_bias"] = nn.Parameter(bias_data.data)

    def _apply_proj(self, x, name):
        """Dispatches computation based on type (Standard, Quant, SVD, etc)"""
        meta_type = self.layer_meta[name]["type"]
        bias = self.params.get(f"{name}_bias", 0)

        if meta_type == "standard":
            return F.linear(x, self.params[f"{name}_weight"]) + bias

        elif meta_type == "quant":
            w = self.params[f"{name}_weight_int8"].float() * self.params[f"{name}_scale"]
            return F.linear(x, w) + bias

        elif meta_type == "svd":
            U, S, Vh = self.params[f"{name}_U"], self.params[f"{name}_S"], self.params[f"{name}_Vh"]
            return ((x @ Vh.mT) * S @ U.mT) + bias

        elif meta_type == "svd_quant":
            U = self.params[f"{name}_U_int8"].float() * self.params[f"{name}_U_scale"]
            Vh = self.params[f"{name}_Vh_int8"].float() * self.params[f"{name}_Vh_scale"]
            S = self.params[f"{name}_S"]
            return ((x @ Vh.mT) * S @ U.mT) + bias

    def forward(self, u):
        batch, seq_len, _ = u.shape

        # 1. Input Projection
        xz = self._apply_proj(u, "in_proj")
        x, z = xz.chunk(2, dim=-1)

        # 2. Convolution
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (D_CONV - 1, 0))
        x_conv = F.conv1d(x_pad, self.params["conv_weight"], bias=self.params.get("conv_bias"), groups=D_INNER)
        x_conv = F.silu(x_conv).transpose(1, 2)

        # 3. SSM Data Dependent Steps (SPLIT VERSION)
        # Instead of one big projection, we do 3 small ones
        dt_rank = self._apply_proj(x_conv, "x_proj_dt")
        B = self._apply_proj(x_conv, "x_proj_b")
        C = self._apply_proj(x_conv, "x_proj_c")

        dt = F.softplus(self._apply_proj(dt_rank, "dt_proj"))

        A = -torch.exp(self.params["A_log"].float())
        D = self.params["D"].float()
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)

        # 4. Recurrence
        h = torch.zeros(batch, D_INNER, D_STATE, device=x.device)
        y_ssm = []
        for t in range(seq_len):
            h = h * dA[:, t] + x_conv[:, t].unsqueeze(-1) * dB[:, t]
            y_t = torch.sum(h * C[:, t].unsqueeze(-2), dim=-1)
            y_t = y_t + (D * x_conv[:, t])
            y_ssm.append(y_t)

        y = torch.stack(y_ssm, dim=1) * F.silu(z)

        # 5. Output Projection
        return self._apply_proj(y, "out_proj")

class MambaModelUnified(nn.Module):
    def __init__(self, hf_model, config, skip_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_layers = skip_layers
        self.embedding = hf_model.backbone.embeddings
        self.final_norm = hf_model.backbone.norm_f
        self.lm_head = hf_model.lm_head

        for i, hf_layer in enumerate(hf_model.backbone.layers):
            layer_cfg = config if i not in skip_layers else {}
            block = UnifiedMambaBlock(i, layer_cfg)
            block.load_weights(hf_layer)
            self.layers.append(block)
            self.norms.append(hf_layer.norm)

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        for layer, norm in zip(self.layers, self.norms):
            x = x + layer(rms_norm(x, norm.weight))
        x = rms_norm(x, self.final_norm.weight)
        return self.lm_head(x)

if __name__ == "__main__":
    from datasets import load_dataset
    import tqdm

    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Loading Wikitext-2 dataset...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

    # --- UPDATED CONFIGURATION ---
    # Now we can target x_proj_dt, x_proj_b, x_proj_c specifically
    mix_config = {
        # Standard large projections: SVD them
        "in_proj":   {"svd_rank": None, "quant": False},
        "out_proj":  {"svd_rank": None, "quant": False},
        
        # Keep dt_proj accurate (no SVD, no Quant) as it is sensitive
        "dt_proj":   {"svd_rank": None, "quant": False}, 
        
        # SPLIT MATRIX CONFIG:
        # 1. dt: Keep full rank (48), just quantize
        "x_proj_dt": {"svd_rank": None, "quant": False}, 
        
        # 2. B: Original Rank is 16. Let's SVD compress to 8 and Quantize.
        "x_proj_b":  {"svd_rank": None,    "quant": True},
        
        # 3. C: Original Rank is 16. Let's SVD compress to 8 and Quantize.
        "x_proj_c":  {"svd_rank": 10,    "quant": False} 
    }
    
    custom_model = MambaModelUnified(hf_model, mix_config, [0, 1, 22, 23]).to(DEVICE)

    # ... [Same eval loop as before] ...
    # (Including the fix for tokenizer and eval loop from previous turn)
    
    def eval_model(model, name):
        nlls = []
        prev_end_loc = 0
        count = 0
        stride = 512
        seq_len = 1024
        max_len = encodings.input_ids.size(1)

        for begin_loc in tqdm.tqdm(range(0, max_len, stride), desc=f"Evaluating {name}"):
            if count > 20: break # Short run for speed

            end_loc = min(begin_loc + seq_len, max_len)
            trg_len = end_loc - prev_end_loc
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()
                loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                nlls.append(loss)
            prev_end_loc = end_loc
            count += 1
            if end_loc == max_len: break
        return torch.exp(torch.stack(nlls).mean())

    print("\nRunning Valid Comparison...")
    ppl_hf = eval_model(hf_model, "HF Original")
    ppl_custom = eval_model(custom_model, "Custom Split-Quant")

    print(f"\nHF: {ppl_hf:.4f} | Custom: {ppl_custom:.4f}")
