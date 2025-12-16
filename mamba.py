import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "state-spaces/mamba-130m-hf"
TEXT_INPUT = "The quick brown fox jumps over the lazy dog"

D_MODEL = 768  # 130M model
EXPAND = 2
D_INNER = D_MODEL * EXPAND
D_STATE = 16
D_CONV = 4
DT_RANK = D_MODEL // 16
N_LAYERS = 24

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(42)


# hugging face does rms_norm so we do it too
def rms_norm(x, weight, eps=1e-5):
    mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
    normed = x * torch.rsqrt(mean_sq + eps)
    return normed * weight


def calculate_perplexity(logits, target_ids):
    shift_logits = logits[:-1, :]
    shift_labels = target_ids[1:]
    loss = F.cross_entropy(shift_logits, shift_labels)
    return torch.exp(loss)


class MambaBlockSVD(nn.Module):
    def __init__(self, layer_idx, svd_config, skip_layers):
        super().__init__()
        self.layer_idx = layer_idx
        self.params = nn.ParameterDict()
        self.use_svd = {k: False for k in ["in_proj", "x_proj", "dt_proj", "out_proj"]}
        self.skip_svd = layer_idx in skip_layers
        self.svd_config = svd_config

    def load_weights(self, hf_layer):
        mixer = hf_layer.mixer

        self.params["conv_weight"] = nn.Parameter(mixer.conv1d.weight.data)
        if mixer.conv1d.bias is not None:
            self.params["conv_bias"] = nn.Parameter(mixer.conv1d.bias.data)

        self.params["A_log"] = nn.Parameter(mixer.A_log.data)
        self.params["D"] = nn.Parameter(mixer.D.data)

        proj_map = {
            "in_proj": mixer.in_proj,
            "x_proj": mixer.x_proj,
            "dt_proj": mixer.dt_proj,
            "out_proj": mixer.out_proj,
        }

        for name, module in proj_map.items():
            weight = module.weight.data
            bias = module.bias.data if module.bias is not None else None
            target_rank = self.svd_config.get(name)

            if target_rank and not self.skip_svd:
                U, S, Vh = torch.linalg.svd(weight, full_matrices=False)
                self.params[f"{name}_U"] = nn.Parameter(U[:, :target_rank])
                self.params[f"{name}_S"] = nn.Parameter(S[:target_rank])
                self.params[f"{name}_Vh"] = nn.Parameter(Vh[:target_rank, :])
                self.use_svd[name] = True
            else:
                self.params[f"{name}_weight"] = nn.Parameter(weight)

            if bias is not None:
                self.params[f"{name}_bias"] = nn.Parameter(bias)

    def _apply_proj(self, x, name):
        if self.use_svd[name]:
            U = self.params[f"{name}_U"]
            S = self.params[f"{name}_S"]
            Vh = self.params[f"{name}_Vh"]

            x_latent = x @ Vh.mT
            x_scaled = x_latent * S
            out = x_scaled @ U.mT
        else:
            w = self.params[f"{name}_weight"]
            out = F.linear(x, w)

        if f"{name}_bias" in self.params:
            out += self.params[f"{name}_bias"]
        return out

    def forward(self, u):
        batch, seq_len, _ = u.shape

        xz = self._apply_proj(u, "in_proj")
        x, z = xz.chunk(2, dim=-1)  # break into [seq_len, 1536]

        # chatmagic to make conv1d work with how weights are organized
        x_t = x.transpose(1, 2)
        x_pad = F.pad(x_t, (D_CONV - 1, 0))

        weight = self.params["conv_weight"]
        bias = self.params.get("conv_bias")

        x_conv = F.conv1d(x_pad, weight, bias=bias, groups=D_INNER)
        x_conv = F.silu(x_conv).transpose(1, 2)

        x_dbl = self._apply_proj(x_conv, "x_proj")
        dt_rank, B, C = torch.split(x_dbl, [DT_RANK, D_STATE, D_STATE], dim=-1)

        dt = F.softplus(self._apply_proj(dt_rank, "dt_proj"))

        A = -torch.exp(self.params["A_log"].float())
        D = self.params["D"].float()  # not sure if we use D

        # I need to understand these pytorch things chat for now
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB = dt.unsqueeze(-1) * B.unsqueeze(-2)

        h = torch.zeros(batch, D_INNER, D_STATE, device=x.device)
        y_ssm = []

        # This runs for each token, not doing Parallel Associative Scan
        for t in range(seq_len):
            h = h * dA[:, t] + x_conv[:, t].unsqueeze(-1) * dB[:, t]
            y_t = torch.sum(h * C[:, t].unsqueeze(-2), dim=-1)
            y_t = y_t + (D * x_conv[:, t])
            y_ssm.append(y_t)

        y_ssm = torch.stack(y_ssm, dim=1)
        y = y_ssm * F.silu(z)

        return self._apply_proj(y, "out_proj")


class MambaModelSVD(nn.Module):
    def __init__(self, hf_model, svd_config, skip_layers):
        super().__init__()
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()

        print(f"SVD Config: {svd_config} | Skipped Layers: {skip_layers}")

        self.embedding = hf_model.backbone.embeddings
        self.final_norm = hf_model.backbone.norm_f
        self.lm_head = hf_model.lm_head

        for i, hf_layer in enumerate(hf_model.backbone.layers):
            block = MambaBlockSVD(i, svd_config, skip_layers)
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
    print(f"Device: {DEVICE}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    hf_model = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    hf_model.eval()

    svd_config = {"in_proj": 768, "out_proj": 768, "x_proj": 60}
    skipped = [0, 1, 2, 21, 22, 23]

    custom_model = MambaModelSVD(hf_model, svd_config, skipped).to(DEVICE)

    inputs = tokenizer(TEXT_INPUT, return_tensors="pt").to(DEVICE)
    input_ids = inputs.input_ids

    with torch.no_grad():
        out_hf = hf_model(input_ids).logits
        ppl_hf = calculate_perplexity(out_hf[0], input_ids[0])

        out_custom = custom_model(input_ids)
        ppl_custom = calculate_perplexity(out_custom[0], input_ids[0])

    diff = (out_custom - out_hf).abs().mean().item()

    print(f"{'Metric':<15} | {'HF Original':<15} | {'Custom SVD':<15}")
    print("-" * 50)
    print(f"{'Perplexity':<15} | {ppl_hf.item():<15.4f} | {ppl_custom.item():<15.4f}")
    print(f"{'Logit Diff':<15} | {'-':<15} | {diff:.6f}")
