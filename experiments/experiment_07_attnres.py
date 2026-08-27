import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Attention Residual
# ============================================================

class AttentionResidual(nn.Module):

    def __init__(self, hidden_dim, num_previous):

        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_previous = num_previous

        # Learnable pseudo-query
        self.query = nn.Parameter(
            torch.randn(hidden_dim)
        )

    def forward(self, representations):

        # representations:
        # [embedding, layer1, layer2, ...]

        num_representations = len(
            representations
        )

        # ----------------------------------------------------
        # Create keys from each representation
        # ----------------------------------------------------

        keys = []

        values = []

        for h in representations:

            # Average across tokens
            key = h.mean(
                dim=1
            )

            keys.append(key)

            values.append(h)

        # [num_layers, batch, hidden]
        keys = torch.stack(keys)

        # ----------------------------------------------------
        # Attention scores
        # ----------------------------------------------------

        scores = torch.einsum(
            "d,lbd->lb",
            self.query,
            keys
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        weights = F.softmax(
            scores,
            dim=0
        )

        # ----------------------------------------------------
        # Weighted combination
        # ----------------------------------------------------

        output = torch.zeros_like(
            representations[0]
        )

        for i, value in enumerate(values):

            weight = weights[i]

            output = (
                output
                + weight[:, None, None]
                * value
            )

        return output, weights


# ============================================================
# Experiment settings
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

batch_size = 1
sequence_length = 32
hidden_dim = 32

num_layers = 6


# ============================================================
# Create fake representations
# ============================================================

representations = []

# Embedding
embedding = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    device=device
)

representations.append(
    embedding
)


# Simulate six layers
current = embedding

for layer in range(num_layers):

    transformation = nn.Linear(
        hidden_dim,
        hidden_dim
    ).to(device)

    current = (
        current
        + transformation(current)
    )

    current = F.layer_norm(
        current,
        (hidden_dim,)
    )

    representations.append(
        current
    )


print()
print(
    "Number of representations:",
    len(representations)
)


# ============================================================
# Attention Residual
# ============================================================

attn_res = AttentionResidual(
    hidden_dim=hidden_dim,
    num_previous=len(representations)
).to(device)


output, weights = attn_res(
    representations
)


# ============================================================
# Print weights
# ============================================================

print()
print("Attention Residual weights:")

for i, weight in enumerate(
    weights[:, 0]
):

    print(
        f"Representation {i}: "
        f"{weight.item():.4f}"
    )


print()
print(
    "Weights sum:",
    weights[:, 0].sum().item()
)


# ============================================================
# Plot attention weights
# ============================================================

plt.figure(figsize=(10, 6))

plt.bar(
    range(len(representations)),
    weights[:, 0]
    .detach()
    .cpu()
    .numpy()
)

plt.xlabel(
    "Representation"
)

plt.ylabel(
    "Attention weight"
)

plt.title(
    "Attention Residual Layer Selection"
)

plt.xticks(
    range(len(representations)),
    [
        "Embedding"
        if i == 0
        else f"Layer {i}"
        for i in range(
            len(representations)
        )
    ],
    rotation=45
)

plt.grid(
    axis="y"
)

plt.tight_layout()

plt.savefig(
    "results/attnres_weights.png",
    dpi=300
)

plt.show()


# ============================================================
# Compare ordinary residual vs AttnRes
# ============================================================

standard_residual = torch.stack(
    representations
).sum(
    dim=0
)


standard_norm = torch.norm(
    standard_residual
).item()

attnres_norm = torch.norm(
    output
).item()


print()
print(
    "Standard residual norm:",
    standard_norm
)

print(
    "Attention residual norm:",
    attnres_norm
)
