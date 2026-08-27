import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# SiTU-GLU
# ============================================================

class SiTUGLU(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        beta=4.0,
        linear_beta=25.0
    ):
        super().__init__()

        self.beta = beta
        self.linear_beta = linear_beta

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

    def forward(self, x):

        # Gate branch
        gate = self.gate_proj(x)

        # Up branch
        up = self.up_proj(x)

        # Compute in FP32 for stability
        gate_fp32 = gate.float()
        up_fp32 = up.float()

        # SiTU gate
        gate_out = (
            self.beta
            * torch.tanh(gate_fp32 / self.beta)
            * torch.sigmoid(gate_fp32)
        )

        # Bounded up branch
        up_out = (
            self.linear_beta
            * torch.tanh(
                up_fp32 / self.linear_beta
            )
        )

        # GLU multiplication
        output = gate_out * up_out

        # Down projection
        output = self.down_proj(
            output.to(x.dtype)
        )

        return output


# ============================================================
# Mini KDA
# ============================================================

class MiniKDA(nn.Module):

    def __init__(
        self,
        hidden_dim,
        head_dim
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.head_dim = head_dim

        # Q K V
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

        # ShortConv
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

        # Alpha
        self.alpha_proj = nn.Linear(
            hidden_dim,
            head_dim
        )

        # Beta
        self.beta_proj = nn.Linear(
            hidden_dim,
            1
        )

        # Output projection
        self.out_proj = nn.Linear(
            head_dim,
            hidden_dim,
            bias=False
        )

    def short_conv(self, x, conv):

        # [B,T,D] -> [B,D,T]
        x = x.transpose(1, 2)

        x = conv(x)

        # Remove future padding
        x = x[:, :, :-2]

        # [B,D,T] -> [B,T,D]
        return x.transpose(1, 2)

    def forward(self, x):

        B, T, _ = x.shape

        # ----------------------------------------------------
        # Q K V
        # ----------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------------------------
        # ShortConv
        # ----------------------------------------------------

        q = self.short_conv(
            q,
            self.q_conv
        )

        k = self.short_conv(
            k,
            self.k_conv
        )

        v = self.short_conv(
            v,
            self.v_conv
        )

        # ----------------------------------------------------
        # Swish / SiLU
        # ----------------------------------------------------

        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        # ----------------------------------------------------
        # L2 normalize Q/K
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Gates
        # ----------------------------------------------------

        alpha = torch.sigmoid(
            self.alpha_proj(x)
        )

        beta = torch.sigmoid(
            self.beta_proj(x)
        )

        # ----------------------------------------------------
        # Recurrent state
        # ----------------------------------------------------

        state = torch.zeros(
            B,
            self.head_dim,
            self.head_dim,
            device=x.device
        )

        outputs = []

        state_norms = []

        I = torch.eye(
            self.head_dim,
            device=x.device
        )

        # ----------------------------------------------------
        # Process sequence
        # ----------------------------------------------------

        for t in range(T):

            batch_outputs = []
            batch_states = []

            for b in range(B):

                S = state[b]

                qt = q[b, t]
                kt = k[b, t]
                vt = v[b, t]

                alpha_t = alpha[b, t]
                beta_t = beta[b, t, 0]

                # Retain old state
                retained = (
                    alpha_t.unsqueeze(1) * S
                )

                # Delta correction
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

                # Write new information
                new_info = (
                    beta_t
                    * torch.outer(
                        kt,
                        vt
                    )
                )

                # New state
                S_new = (
                    corrected
                    + new_info
                )

                # Read
                output = S_new.T @ qt

                batch_states.append(S_new)
                batch_outputs.append(output)

            state = torch.stack(
                batch_states
            )

            output = torch.stack(
                batch_outputs
            )

            outputs.append(output)

            state_norms.append(
                torch.norm(state).item()
            )

        outputs = torch.stack(
            outputs
        )

        outputs = outputs.transpose(
            0,
            1
        )

        # Project back to hidden dimension
        outputs = self.out_proj(
            outputs
        )

        return outputs, state_norms


# ============================================================
# Mini K3 Block
# ============================================================

class MiniK3Block(nn.Module):

    def __init__(
        self,
        hidden_dim=32,
        head_dim=16,
        intermediate_dim=64
    ):
        super().__init__()

        self.kda = MiniKDA(
            hidden_dim,
            head_dim
        )

        self.situ = SiTUGLU(
            hidden_dim,
            intermediate_dim
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

    def forward(self, x):

        # KDA
        kda_output, state_norms = self.kda(
            x
        )

        # Residual
        x = x + kda_output

        # Normalize
        x_norm = self.norm(x)

        # SiTU-GLU
        mlp_output = self.situ(
            x_norm
        )

        # Residual
        output = x + mlp_output

        return output, state_norms


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


# Small dimensions
batch_size = 1
sequence_length = 32
hidden_dim = 32
head_dim = 16
intermediate_dim = 64


# Random tokens/embeddings
x = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    device=device
)


# Create block
model = MiniK3Block(
    hidden_dim=hidden_dim,
    head_dim=head_dim,
    intermediate_dim=intermediate_dim
).to(device)


# Forward
output, state_norms = model(x)


# ============================================================
# Results
# ============================================================

print()
print("Input shape :", x.shape)
print("Output shape:", output.shape)

print()
print("Input mean  :", x.mean().item())
print("Output mean :", output.mean().item())

print()
print(
    "Input std   :",
    x.std().item()
)

print(
    "Output std  :",
    output.std().item()
)

print()
print(
    "Final KDA state norm:",
    state_norms[-1]
)


# ============================================================
# Plot KDA state
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, sequence_length + 1),
    state_norms,
    marker="o"
)

plt.xlabel("Token")
plt.ylabel("||S||")

plt.title(
    "Mini K3 Block - KDA State Evolution"
)

plt.grid(True)

plt.savefig(
    "results/mini_k3_kda_state.png",
    dpi=300
)

plt.show()


# ============================================================
# Compare input/output magnitude
# ============================================================

input_norm = torch.norm(
    x,
    dim=-1
)[0].detach().cpu().numpy()

output_norm = torch.norm(
    output,
    dim=-1
)[0].detach().cpu().numpy()


plt.figure(figsize=(10, 6))

plt.plot(
    input_norm,
    label="Input"
)

plt.plot(
    output_norm,
    label="Output"
)

plt.xlabel("Token")
plt.ylabel("Vector norm")

plt.title(
    "Mini K3 Block - Input vs Output"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/mini_k3_input_output.png",
    dpi=300
)

plt.show()
