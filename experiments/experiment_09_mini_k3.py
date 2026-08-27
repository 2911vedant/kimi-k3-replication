import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# SiTU-GLU
# ============================================================

class SiTUGLU(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()

        self.gate_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=False
        )

        self.up_proj = nn.Linear(
            hidden_dim,
            intermediate_dim,
            bias=False
        )

        self.down_proj = nn.Linear(
            intermediate_dim,
            hidden_dim,
            bias=False
        )

        self.beta = 4.0
        self.linear_beta = 25.0

    def forward(self, x):

        gate = self.gate_proj(x)
        up = self.up_proj(x)

        gate = gate.float()
        up = up.float()

        gate = (
            self.beta
            * torch.tanh(gate / self.beta)
            * torch.sigmoid(gate)
        )

        up = (
            self.linear_beta
            * torch.tanh(
                up / self.linear_beta
            )
        )

        output = gate * up

        return self.down_proj(
            output.to(x.dtype)
        )


# ============================================================
# Mini KDA
# ============================================================

class MiniKDA(nn.Module):

    def __init__(self, hidden_dim, head_dim):
        super().__init__()

        self.head_dim = head_dim

        self.q_proj = nn.Linear(
            hidden_dim,
            head_dim,
            bias=False
        )

        self.k_proj = nn.Linear(
            hidden_dim,
            head_dim,
            bias=False
        )

        self.v_proj = nn.Linear(
            hidden_dim,
            head_dim,
            bias=False
        )

        self.q_conv = nn.Conv1d(
            head_dim,
            head_dim,
            3,
            padding=2,
            groups=head_dim
        )

        self.k_conv = nn.Conv1d(
            head_dim,
            head_dim,
            3,
            padding=2,
            groups=head_dim
        )

        self.v_conv = nn.Conv1d(
            head_dim,
            head_dim,
            3,
            padding=2,
            groups=head_dim
        )

        self.alpha_proj = nn.Linear(
            hidden_dim,
            head_dim
        )

        self.beta_proj = nn.Linear(
            hidden_dim,
            1
        )

        self.out_proj = nn.Linear(
            head_dim,
            hidden_dim,
            bias=False
        )

    def short_conv(self, x, conv):

        x = x.transpose(1, 2)
        x = conv(x)
        x = x[:, :, :-2]

        return x.transpose(1, 2)

    def forward(self, x):

        B, T, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self.short_conv(q, self.q_conv)
        k = self.short_conv(k, self.k_conv)
        v = self.short_conv(v, self.v_conv)

        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        q = F.normalize(
            q,
            p=2,
            dim=-1
        )

        k = F.normalize(
            k,
            p=2,
            dim=-1
        )

        alpha = torch.sigmoid(
            self.alpha_proj(x)
        )

        beta = torch.sigmoid(
            self.beta_proj(x)
        )

        state = torch.zeros(
            B,
            self.head_dim,
            self.head_dim,
            device=x.device
        )

        outputs = []

        I = torch.eye(
            self.head_dim,
            device=x.device
        )

        for t in range(T):

            batch_states = []
            batch_outputs = []

            for b in range(B):

                S = state[b]

                qt = q[b, t]
                kt = k[b, t]
                vt = v[b, t]

                alpha_t = alpha[b, t]
                beta_t = beta[b, t, 0]

                retained = (
                    alpha_t.unsqueeze(1)
                    * S
                )

                correction = (
                    I
                    - beta_t
                    * torch.outer(
                        kt,
                        kt
                    )
                )

                corrected = (
                    correction @ retained
                )

                new_info = (
                    beta_t
                    * torch.outer(
                        kt,
                        vt
                    )
                )

                S_new = (
                    corrected
                    + new_info
                )

                output = S_new.T @ qt

                batch_states.append(S_new)
                batch_outputs.append(output)

            state = torch.stack(
                batch_states
            )

            outputs.append(
                torch.stack(batch_outputs)
            )

        outputs = torch.stack(outputs)

        outputs = outputs.transpose(0, 1)

        return self.out_proj(outputs)


# ============================================================
# Tiny Expert
# ============================================================

class Expert(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_dim,
            intermediate_dim
        )

        self.fc2 = nn.Linear(
            intermediate_dim,
            hidden_dim
        )

    def forward(self, x):

        return self.fc2(
            F.silu(
                self.fc1(x)
            )
        )


# ============================================================
# Tiny Sparse MoE
# ============================================================

class TinyMoE(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts=8,
        top_k=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(
            hidden_dim,
            num_experts,
            bias=False
        )

        self.experts = nn.ModuleList(
            [
                Expert(
                    hidden_dim,
                    intermediate_dim
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):

        probabilities = F.softmax(
            self.router(x),
            dim=-1
        )

        top_values, top_indices = torch.topk(
            probabilities,
            self.top_k,
            dim=-1
        )

        top_weights = (
            top_values
            / (
                top_values.sum(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        output = torch.zeros_like(x)

        for expert_id, expert in enumerate(
            self.experts
        ):

            expert_output = expert(x)

            for k in range(self.top_k):

                mask = (
                    top_indices[:, :, k]
                    == expert_id
                )

                weight = top_weights[:, :, k]

                output = output + (
                    expert_output
                    * mask.unsqueeze(-1)
                    * weight.unsqueeze(-1)
                )

        return output, probabilities


# ============================================================
# Attention Residual
# ============================================================

class AttentionResidual(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.query = nn.Parameter(
            torch.randn(hidden_dim)
        )

    def forward(self, representations):

        keys = []

        for h in representations:

            keys.append(
                h.mean(dim=1)
            )

        keys = torch.stack(keys)

        scores = torch.einsum(
            "d,lbd->lb",
            self.query,
            keys
        )

        weights = F.softmax(
            scores,
            dim=0
        )

        output = torch.zeros_like(
            representations[0]
        )

        for i, h in enumerate(
            representations
        ):

            output = output + (
                weights[i][:, None, None]
                * h
            )

        return output, weights


# ============================================================
# Mini K3 Layer
# ============================================================

class MiniK3Layer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        head_dim,
        intermediate_dim
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(
            hidden_dim
        )

        self.kda = MiniKDA(
            hidden_dim,
            head_dim
        )

        self.norm2 = nn.LayerNorm(
            hidden_dim
        )

        self.moe = TinyMoE(
            hidden_dim,
            intermediate_dim
        )

    def forward(self, x):

        # KDA
        kda_out = self.kda(
            self.norm1(x)
        )

        x = x + kda_out

        # MoE
        moe_out, router_probs = self.moe(
            self.norm2(x)
        )

        x = x + moe_out

        return x, router_probs


# ============================================================
# MINI K3
# ============================================================

class MiniK3(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim=32,
        head_dim=16,
        intermediate_dim=64,
        num_layers=4
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

        self.layers = nn.ModuleList(
            [
                MiniK3Layer(
                    hidden_dim,
                    head_dim,
                    intermediate_dim
                )
                for _ in range(num_layers)
            ]
        )

        # Attention Residual controller
        self.attn_res = AttentionResidual(
            hidden_dim
        )

        self.final_norm = nn.LayerNorm(
            hidden_dim
        )

        self.lm_head = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        representations = [x]

        router_outputs = []

        for layer in self.layers:

            x, router_probs = layer(x)

            representations.append(x)

            router_outputs.append(
                router_probs
            )

        # Attention Residual
        x, residual_weights = self.attn_res(
            representations
        )

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return (
            logits,
            residual_weights,
            router_outputs
        )


# ============================================================
# Experiment
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ------------------------------------------------------------
# Small model configuration
# ------------------------------------------------------------

vocab_size = 100
sequence_length = 32

model = MiniK3(
    vocab_size=vocab_size,
    hidden_dim=32,
    head_dim=16,
    intermediate_dim=64,
    num_layers=4
).to(device)


# ------------------------------------------------------------
# Random token sequence
# ------------------------------------------------------------

tokens = torch.randint(
    0,
    vocab_size,
    (
        1,
        sequence_length
    ),
    device=device
)


# ------------------------------------------------------------
# Forward pass
# ------------------------------------------------------------

logits, residual_weights, router_outputs = model(
    tokens
)


# ============================================================
# Print results
# ============================================================

print()
print("Token shape :", tokens.shape)
print("Logits shape:", logits.shape)

print()
print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)

print()
print("Attention Residual weights:")

for i, weight in enumerate(
    residual_weights[:, 0]
):

    print(
        f"Representation {i}: "
        f"{weight.item():.4f}"
    )

print()
print(
    "Residual weight sum:",
    residual_weights[:, 0].sum().item()
)


# ============================================================
# Router analysis
# ============================================================

print()
print("Average router probabilities:")

average_router = torch.stack(
    [
        r.mean(dim=(0, 1))
        for r in router_outputs
    ]
)

for layer in range(
    average_router.shape[0]
):

    print(
        f"Layer {layer + 1}:",
        average_router[layer]
        .detach()
        .cpu()
        .numpy()
    )


# ============================================================
# Plot Attention Residuals
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    range(
        len(residual_weights)
    ),
    residual_weights[:, 0]
    .detach()
    .cpu()
    .numpy()
)

plt.xlabel("Representation")
plt.ylabel("Weight")

plt.title(
    "Mini K3 Attention Residual Weights"
)

plt.xticks(
    range(
        len(residual_weights)
    ),
    [
        "Embedding"
        if i == 0
        else f"Layer {i}"
        for i in range(
            len(residual_weights)
        )
    ],
    rotation=45
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "results/mini_k3_attnres.png",
    dpi=300
)

plt.show()


# ============================================================
# Plot router probabilities
# ============================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    average_router
    .detach()
    .cpu()
    .numpy(),
    aspect="auto"
)

plt.xlabel("Expert")
plt.ylabel("K3 Layer")

plt.title(
    "Mini K3 MoE Router Distribution"
)

plt.colorbar(
    label="Probability"
)

plt.tight_layout()

plt.savefig(
    "results/mini_k3_router.png",
    dpi=300
)

plt.show()


print()
print("===================================")
print("MINI K3 FORWARD PASS SUCCESSFUL")
print("===================================")
