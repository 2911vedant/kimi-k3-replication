import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


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

        x = self.fc1(x)

        x = F.silu(x)

        x = self.fc2(x)

        return x


# ============================================================
# Tiny Latent MoE
# ============================================================

class TinyMoE(nn.Module):

    def __init__(
        self,
        hidden_dim=32,
        intermediate_dim=64,
        num_experts=8,
        top_k=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        # ----------------------------------------------------
        # Router
        # ----------------------------------------------------

        self.router = nn.Linear(
            hidden_dim,
            num_experts,
            bias=False
        )

        # ----------------------------------------------------
        # Experts
        # ----------------------------------------------------

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

        B, T, D = x.shape

        # ----------------------------------------------------
        # Router logits
        # ----------------------------------------------------

        logits = self.router(x)

        # ----------------------------------------------------
        # Router probabilities
        # ----------------------------------------------------

        probabilities = F.softmax(
            logits,
            dim=-1
        )

        # ----------------------------------------------------
        # Select top-k experts
        # ----------------------------------------------------

        top_values, top_indices = torch.topk(
            probabilities,
            self.top_k,
            dim=-1
        )

        # Normalize selected weights
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

        # ----------------------------------------------------
        # Expert outputs
        # ----------------------------------------------------

        output = torch.zeros_like(x)

        for expert_id, expert in enumerate(
            self.experts
        ):

            expert_output = expert(x)

            # Which tokens selected this expert?
            mask = (
                top_indices == expert_id
            )

            for k in range(self.top_k):

                selected = mask[:, :, k]

                weight = top_weights[
                    :, :, k
                ]

                output = output + (
                    expert_output
                    * selected.unsqueeze(-1)
                    * weight.unsqueeze(-1)
                )

        return output, probabilities, top_indices


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
# Settings
# ------------------------------------------------------------

batch_size = 1
sequence_length = 32
hidden_dim = 32

num_experts = 8
top_k = 2


# ------------------------------------------------------------
# Input
# ------------------------------------------------------------

x = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    device=device
)


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

moe = TinyMoE(
    hidden_dim=hidden_dim,
    intermediate_dim=64,
    num_experts=num_experts,
    top_k=top_k
).to(device)


# ------------------------------------------------------------
# Forward
# ------------------------------------------------------------

output, probabilities, top_indices = moe(x)


# ============================================================
# Results
# ============================================================

print()
print("Input shape :", x.shape)
print("Output shape:", output.shape)

print()
print("Number of experts:", num_experts)
print("Experts selected per token:", top_k)


# ============================================================
# Expert usage
# ============================================================

usage = torch.zeros(
    num_experts
)

for expert_id in range(num_experts):

    usage[expert_id] = (
        top_indices == expert_id
    ).sum().item()


print()
print("Expert usage:")

for expert_id in range(num_experts):

    print(
        f"Expert {expert_id}: "
        f"{int(usage[expert_id].item())} selections"
    )


# ============================================================
# Plot expert usage
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    range(num_experts),
    usage.numpy()
)

plt.xlabel("Expert")
plt.ylabel("Number of token selections")

plt.title(
    "Tiny MoE Expert Usage"
)

plt.xticks(
    range(num_experts)
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "results/moe_expert_usage.png",
    dpi=300
)

plt.show()


# ============================================================
# Plot routing probabilities
# ============================================================

plt.figure(figsize=(10, 6))

plt.imshow(
    probabilities[0]
    .detach()
    .cpu()
    .numpy(),
    aspect="auto"
)

plt.xlabel("Expert")
plt.ylabel("Token")

plt.title(
    "MoE Router Probabilities"
)

plt.colorbar(
    label="Probability"
)

plt.tight_layout()

plt.savefig(
    "results/moe_router_heatmap.png",
    dpi=300
)

plt.show()


# ============================================================
# Check top-k routing
# ============================================================

print()
print("First 5 tokens:")

for token in range(5):

    experts = top_indices[
        0,
        token
    ].tolist()

    weights = top_weights = probabilities[
        0,
        token
    ][experts].tolist()

    print(
        f"Token {token + 1}: "
        f"experts={experts}, "
        f"weights={weights}"
    )
