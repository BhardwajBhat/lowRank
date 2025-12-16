import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# --- Configuration & Hyperparameters ---
MODEL_ID = "state-spaces/mamba-130m-hf"
TEXT_INPUT = "The quick brown fox jumps over the lazy dog"

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
    """Root Mean Square Normalization"""
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(mean_sq + eps)
    return normed * weight


def calculate_perplexity(logits, target_ids):
    """Calculates perplexity (exp of cross entropy)"""
    shift_logits = logits[:-1, :]
    shift_labels = target_ids[1:]
    loss = F.cross_entropy(shift_logits, shift_labels)
    return torch.exp(loss)


# --- The Core Logic: Unified Mamba Block ---


class UnifiedMambaBlock(nn.Module):
    def __init__(self, layer_idx, config):
        """
        Args:
            layer_idx: Index of the layer (0 to 23)
            config: Dict defining how to treat each projection.
                    Example: {"in_proj": {"svd_rank": 128, "quant": True}}
        """
        super().__init__()
        self.layer_idx = layer_idx
        self.config = config
        self.params = nn.ParameterDict()

        # Metadata to track what operations to run for each sub-layer
        self.layer_meta = {
            k: {"type": "standard"}
            for k in ["in_proj", "x_proj", "dt_proj", "out_proj"]
        }

    def _quantize(self, tensor):
        """Helper: Compresses FP32 -> Int8 + Scale (AbsMax Symmetric)"""
        max_val = tensor.abs().max()
        scale = max_val / 127.0
        # Clamp to ensure no overflow errors
        int8_val = (tensor / scale).round().clamp(-127, 127).to(torch.int8)
        return int8_val, scale

    def load_weights(self, hf_layer):
        mixer = hf_layer.mixer

        # --- 1. Load Static SSM Params (Always FP32 for stability) ---
        self.params["conv_weight"] = nn.Parameter(mixer.conv1d.weight.data)
        if mixer.conv1d.bias is not None:
            self.params["conv_bias"] = nn.Parameter(mixer.conv1d.bias.data)
        self.params["A_log"] = nn.Parameter(mixer.A_log.data)
        self.params["D"] = nn.Parameter(mixer.D.data)

        # --- 2. Process Projections (SVD / Quant / Standard) ---
        proj_map = {
            "in_proj": mixer.in_proj,
            "x_proj": mixer.x_proj,
            "dt_proj": mixer.dt_proj,
            "out_proj": mixer.out_proj,
        }

        for name, module in proj_map.items():
            W = module.weight.data
            bias = module.bias.data if module.bias is not None else None

            # Get config for this specific projection
            cfg = self.config.get(name, {})
            rank = cfg.get("svd_rank", None)
            do_quant = cfg.get("quant", False)

            # --- BRANCH 1: SVD ENABLED ---
            if rank is not None:
                U, S, Vh = torch.linalg.svd(W, full_matrices=False)
                U = U[:, :rank]
                S = S[:rank]
                Vh = Vh[:rank, :]

                if do_quant:
                    # [Hybrid: SVD + Quantization]
                    # Quantize U and Vh matrices. Keep S as float.
                    u_int8, u_scale = self._quantize(U)
                    vh_int8, vh_scale = self._quantize(Vh)

                    self.params[f"{name}_U_int8"] = nn.Parameter(
                        u_int8, requires_grad=False
                    )
                    self.params[f"{name}_U_scale"] = nn.Parameter(
                        u_scale, requires_grad=False
                    )
                    self.params[f"{name}_Vh_int8"] = nn.Parameter(
                        vh_int8, requires_grad=False
                    )
                    self.params[f"{name}_Vh_scale"] = nn.Parameter(
                        vh_scale, requires_grad=False
                    )
                    self.params[f"{name}_S"] = nn.Parameter(S)
                    self.layer_meta[name]["type"] = "svd_quant"
                else:
                    # [SVD Only]
                    self.params[f"{name}_U"] = nn.Parameter(U)
                    self.params[f"{name}_S"] = nn.Parameter(S)
                    self.params[f"{name}_Vh"] = nn.Parameter(Vh)
                    self.layer_meta[name]["type"] = "svd"

            # --- BRANCH 2: NO SVD (Standard Matrix) ---
            else:
                if do_quant:
                    # [Quantization Only]
                    w_int8, w_scale = self._quantize(W)
                    self.params[f"{name}_weight_int8"] = nn.Parameter(
                        w_int8, requires_grad=False
                    )
                    self.params[f"{name}_scale"] = nn.Parameter(
                        w_scale, requires_grad=False
                    )
                    self.layer_meta[name]["type"] = "quant"
                else:
                    # [Standard Linear]
                    self.params[f"{name}_weight"] = nn.Parameter(W)
                    self.layer_meta[name]["type"] = "standard"

            if bias is not None:
                self.params[f"{name}_bias"] = nn.Parameter(bias)

    def _apply_proj(self, x, name):
        """Dispatches the computation based on the stored type."""
        meta_type = self.layer_meta[name]["type"]
        bias = self.params.get(f"{name}_bias", 0)

        # 1. Standard Linear
        if meta_type == "standard":
            return F.linear(x, self.params[f"{name}_weight"]) + bias

        # 2. Quantized Linear (De-quantize -> Multiply)
        elif meta_type == "quant":
            w = (
                self.params[f"{name}_weight_int8"].float()
                * self.params[f"{name}_scale"]
            )
            return F.linear(x, w) + bias

        # 3. SVD Linear (Low Rank Approximation)
        elif meta_type == "svd":
            U = self.params[f"{name}_U"]
            S = self.params[f"{name}_S"]
            Vh = self.params[f"{name}_Vh"]

            # x @ Vh.T @ S @ U.T
            x_latent = x @ Vh.mT
            x_scaled = x_latent * S
            return (x_scaled @ U.mT) + bias

        # 4. SVD + Quantized (Hybrid)
        elif meta_type == "svd_quant":
            U = self.params[f"{name}_U_int8"].float() * self.params[f"{name}_U_scale"]
            Vh = (
                self.params[f"{name}_Vh_int8"].float() * self.params[f"{name}_Vh_scale"]
            )
            S = self.params[f"{name}_S"]

            x_latent = x @ Vh.mT
            x_scaled = x_latent * S
            return (x_scaled @ U.mT) + bias

    def forward(self, u):
        batch, seq_len, _ = u.shape

        # --- 1. Input Projection ---
        xz = self._apply_proj(u, "in_proj")
        x, z = xz.chunk(2, dim=-1)

        # --- 2. 1D Convolution ---
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (D_CONV - 1, 0))  # Causal padding

        weight = self.params["conv_weight"]
        bias = self.params.get("conv_bias")

        x_conv = F.conv1d(x_pad, weight, bias=bias, groups=D_INNER)
        x_conv = F.silu(x_conv).transpose(1, 2)

        # --- 3. SSM Data Dependent Steps ---
        x_dbl = self._apply_proj(x_conv, "x_proj")
        dt_rank, B, C = torch.split(x_dbl, [DT_RANK, D_STATE, D_STATE], dim=-1)

        dt = F.softplus(self._apply_proj(dt_rank, "dt_proj"))

        A = -torch.exp(self.params["A_log"].float())
        D = self.params["D"].float()

        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)

        # --- 4. SSM Recurrence (Manual Loop) ---
        h = torch.zeros(batch, D_INNER, D_STATE, device=x.device)
        y_ssm = []

        for t in range(seq_len):
            h = h * dA[:, t] + x_conv[:, t].unsqueeze(-1) * dB[:, t]
            y_t = torch.sum(h * C[:, t].unsqueeze(-2), dim=-1)
            y_t = y_t + (D * x_conv[:, t])
            y_ssm.append(y_t)

        y_ssm = torch.stack(y_ssm, dim=1)
        y = y_ssm * F.silu(z)

        # --- 5. Output Projection ---
        return self._apply_proj(y, "out_proj")


class MambaModelUnified(nn.Module):
    def __init__(self, hf_model, config, skip_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.skip_layers = skip_layers

        # Copy Embeddings and Head
        self.embedding = hf_model.backbone.embeddings
        self.final_norm = hf_model.backbone.norm_f
        self.lm_head = hf_model.lm_head

        # Reconstruct Layers
        for i, hf_layer in enumerate(hf_model.backbone.layers):
            # If layer is skipped, pass empty config (Standard mode)
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


# # --- Execution ---

# if __name__ == "__main__":
#     print(f"Device: {DEVICE}")

#     # 1. Load Original Model (Teacher)
#     tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
#     hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
#     hf_model.eval()

#     # 2. Define Compression Strategy
#     # Here we mix different strategies to test the Unified Block flexibility
#     mix_config = {
#         # in_proj: Hybrid (SVD Rank 256 + Int8 Quantization)
#         "in_proj":  {"svd_rank": None, "quant": True},

#         # out_proj: Quantization Only (Full Rank, Int8)
#         "out_proj": {"svd_rank": None, "quant": True},

#         # x_proj: SVD Only (Low Rank 32, FP32)
#         "x_proj":   {"svd_rank": None,  "quant": False},

#         # dt_proj: Standard (No compression, sensitive layer)
#         "dt_proj":  {"svd_rank": None, "quant": False}
#     }

#     # We typically skip compression on the first and last few layers to preserve accuracy
#     skipped_layers = [0, 1, 22, 23]

#     print(f"\nCompression Config:\n{mix_config}")
#     print(f"Skipping Layers: {skipped_layers}")

#     # 3. Create Custom Model
#     custom_model = MambaModelUnified(hf_model, mix_config, skipped_layers).to(DEVICE)

#     # 4. Prepare Input
#     inputs = tokenizer(TEXT_INPUT, return_tensors="pt").to(DEVICE)
#     input_ids = inputs.input_ids

#     # 5. Run Comparison
#     print("\nRunning Inference...")
#     with torch.no_grad():
#         # Run HF Model
#         out_hf = hf_model(input_ids).logits
#         ppl_hf = calculate_perplexity(out_hf[0], input_ids[0])

#         # Run Custom Unified Model
#         out_custom = custom_model(input_ids)
#         ppl_custom = calculate_perplexity(out_custom[0], input_ids[0])

#     diff = (out_custom - out_hf).abs().mean().item()

#     # 6. Report Results
#     print("\n" + "=" * 60)
#     print(f"{'Metric':<20} | {'HF Original':<15} | {'Custom Unified':<15}")
#     print("-" * 60)
#     print(f"{'Perplexity':<20} | {ppl_hf.item():<15.4f} | {ppl_custom.item():<15.4f}")
#     print(f"{'Logit Diff (Mean)':<20} | {'-':<15} | {diff:.6f}")
#     print("=" * 60)

# Replace the "Execution" block in your script with this:

if __name__ == "__main__":
    from datasets import load_dataset
    import tqdm

    print(f"Device: {DEVICE}")

    # 1. Load Data (Wikitext-2 Test Split)
    print("Loading Wikitext-2 dataset...")
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    # Filter for reasonable length samples to save time
    encodings = tokenizer("\n\n".join(test_data["text"]), return_tensors="pt")

    # Sliding window settings
    stride = 512
    seq_len = 1024
    max_len = encodings.input_ids.size(1)

    # 2. setup Models (Same as your code)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE).eval()

    mix_config = {
        "in_proj": {"svd_rank": None, "quant": True},
        "out_proj": {"svd_rank": None, "quant": True},
        "x_proj": {"svd_rank": None, "quant": False},
        "dt_proj": {"svd_rank": None, "quant": False},
    }
    custom_model = MambaModelUnified(hf_model, mix_config, [0, 1, 22, 23]).to(DEVICE)

    # 3. proper Loop
    def eval_model(model, name):
        nlls = []
        prev_end_loc = 0
        count = 0
        # Run just 50 chunks for a quick valid test
        for begin_loc in tqdm.tqdm(
            range(0, max_len, stride), desc=f"Evaluating {name}"
        ):
            if count > 50:
                break

            end_loc = min(begin_loc + seq_len, max_len)
            trg_len = (
                end_loc - prev_end_loc
            )  # may be different from stride on last loop

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100  # Ignore context in loss

            with torch.no_grad():
                outputs = model(input_ids)
                # --- FIX START ---
                # Check if outputs is an object (HF) or just a Tensor (Custom)
                if hasattr(outputs, "logits"):
                    logits = outputs.logits
                else:
                    logits = outputs  # Your custom model returns this directl
                # Calculate loss manually to handle shifting correctly
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = target_ids[..., 1:].contiguous()

                loss = F.cross_entropy(
                    shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
                )
                nlls.append(loss)

            prev_end_loc = end_loc
            count += 1
            if end_loc == max_len:
                break

        return torch.exp(torch.stack(nlls).mean())

    print("\nRunning Valid Comparison...")
    ppl_hf = eval_model(hf_model, "HF Original")
    ppl_custom = eval_model(custom_model, "Custom Unified")

    print("\n" + "=" * 60)
    print(f"{'Metric':<20} | {'HF Original':<15} | {'Custom Unified':<15}")
    print("-" * 60)
    print(f"{'Perplexity':<20} | {ppl_hf.item():<15.4f} | {ppl_custom.item():<15.4f}")
    print("=" * 60)
