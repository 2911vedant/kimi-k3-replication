import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ==========================================
# KDA Projection Block
# ==========================================

class KDAProjection(nn.Module):

    def __init__(self, input_dim, head_dim, kernel_size=3):
        super().__init__()

        self.q_proj = nn.Linear(input_dim, head_dim)
        self.k_proj = nn.Linear(input_dim, head_dim)
        self.v_proj = nn.Linear(input_dim, head_dim)

        # Simple causal-style short convolution
        self.q_conv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size,
            padding=kernel_size - 1,
            groups=head_dim
        )

        self.k_conv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size,
            padding=kernel_size - 1,
            groups=head_dim
        )

        self.v_conv = nn.Conv1d(
            head_dim,
            head_dim,
            kernel_size,
            padding=kernel_size - 1,
            groups=head_dim
        )

    def short_conv(self, x, conv):

        # x: [batch, sequence, dimension]

        x = x.transpose(1, 2)

        x = conv(x)

        # Remove future positions
        x = x[:, :, :x.shape[2] - (conv.kernel_size[0] - 1)]

        x = x.transpose(1, 2)

        return x

    def forward(self, x):

        # ----------------------------------
        # Linear projections
        # ----------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------
        # Short convolution
        # ----------------------------------

        q = self.short_conv(q, self.q_conv)
        k = self.short_conv(k, self.k_conv)
        v = self.short_conv(v, self.v_conv)

        # ----------------------------------
        # Swish activation
        # ----------------------------------

        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        # ----------------------------------
        # L2 normalization
        # ----------------------------------

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        return q, k, v


# ==========================================
# Settings
# ==========================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

batch_size = 1
sequence_length = 32
input_dim = 32
head_dim = 16


# ==========================================
# Create random input
# ==========================================

x = torch.randn(
    batch_size,
    sequence_length,
    input_dim,
    device=device
)


# ==========================================
# Create KDA projection block
# ==========================================

projection = KDAProjection(
    input_dim,
    head_dim
).to(device)


# ==========================================
# Generate Q, K, V
# ==========================================

q, k, v = projection(x)


print()
print("Input shape :", x.shape)
print("Q shape     :", q.shape)
print("K shape     :", k.shape)
print("V shape     :", v.shape)


# ==========================================
# Check Q/K normalization
# ==========================================

q_norms = torch.norm(q, dim=-1)
k_norms = torch.norm(k, dim=-1)

print()
print("Average Q norm:", q_norms.mean().item())
print("Average K norm:", k_norms.mean().item())


# ==========================================
# Plot Q/K norms
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    q_norms[0].detach().cpu().numpy(),
    label="Q norm"
)

plt.plot(
    k_norms[0].detach().cpu().numpy(),
    label="K norm"
)

plt.axhline(
    1.0,
    linestyle="--",
    label="Target norm = 1"
)

plt.xlabel("Token")
plt.ylabel("L2 norm")

plt.title("KDA Q/K L2 Normalization")

plt.legend()
plt.grid(True)

plt.savefig(
    "results/kda_qk_normalization.png",
    dpi=300
)

plt.show()
