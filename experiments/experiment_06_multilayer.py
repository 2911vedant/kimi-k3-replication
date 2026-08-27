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

        state_norms = []

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

                # Retention
                retained = (
                    alpha_t.unsqueeze(1)
                    * S
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

        outputs = torch.stack(outputs)

        outputs = outputs.transpose(
            0,
            1
        )

        outputs = self.out_proj(outputs)

        return outputs, state_norms


# ============================================================
# One Mini-K3 Block
# ============================================================

class MiniK3Block(nn.Module):

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

        self.situ = SiTUGLU(
            hidden_dim,
            intermediate_dim
        )

    def forward(self, x):

        # ----------------------------
        # KDA sub-layer
        # ----------------------------

        kda_input = self.norm1(x)

        kda_output, state_norms = self.kda(
            kda_input
        )

        x = x + kda_output

        # ----------------------------
        # SiTU-GLU sub-layer
        # ----------------------------

        mlp_input = self.norm2(x)

        mlp_output = self.situ(
            mlp_input
        )

        x = x + mlp_output

        return x, state_norms


# ============================================================
# Multi-Layer Mini K3
# ============================================================

class MiniK3(nn.Module):

    def __init__(
        self,
        num_layers=4,
        hidden_dim=32,
        head_dim=16,
        intermediate_dim=64
    ):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                MiniK3Block(
                    hidden_dim,
                    head_dim,
                    intermediate_dim
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x):

        layer_outputs = []
        layer_state_norms = []

        for i, layer in enumerate(
            self.layers
        ):

            x, state_norms = layer(x)

            layer_outputs.append(
                x
            )

            layer_state_norms.append(
                state_norms
            )

        return (
            x,
            layer_outputs,
            layer_state_norms
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


# ============================================================
# Model settings
# ============================================================

batch_size = 1
sequence_length = 32

hidden_dim = 32
head_dim = 16
intermediate_dim = 64

num_layers = 4


# ============================================================
# Random input
# ============================================================

x = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    device=device
)


# ============================================================
# Create model
# ============================================================

model = MiniK3(
    num_layers=num_layers,
    hidden_dim=hidden_dim,
    head_dim=head_dim,
    intermediate_dim=intermediate_dim
).to(device)


# ============================================================
# Forward pass
# ============================================================

output, layer_outputs, layer_state_norms = model(x)


# ============================================================
# Print information
# ============================================================

print()
print("Number of layers:", num_layers)

print(
    "Input shape :",
    x.shape
)

print(
    "Output shape:",
    output.shape
)

print()

for i, layer_output in enumerate(
    layer_outputs
):

    print(
        f"Layer {i + 1} output shape:",
        layer_output.shape
    )


# ============================================================
# Calculate layer norms
# ============================================================

layer_norms = []

for layer_output in layer_outputs:

    norm = torch.norm(
        layer_output,
        dim=-1
    ).mean().item()

    layer_norms.append(norm)


print()

for i, value in enumerate(
    layer_norms
):

    print(
        f"Layer {i + 1} average vector norm:",
        value
    )


# ============================================================
# Plot layer output magnitude
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, num_layers + 1),
    layer_norms,
    marker="o"
)

plt.xlabel("Layer")
plt.ylabel("Average vector norm")

plt.title(
    "Mini K3 - Representation Through Layers"
)

plt.grid(True)

plt.savefig(
    "results/mini_k3_layer_norms.png",
    dpi=300
)

plt.show()


# ============================================================
# Plot KDA state for every layer
# ============================================================

plt.figure(figsize=(10, 6))

for i in range(num_layers):

    plt.plot(
        range(
            1,
            sequence_length + 1
        ),
        layer_state_norms[i],
        label=f"Layer {i + 1}"
    )

plt.xlabel("Token")
plt.ylabel("||S||")

plt.title(
    "Mini K3 - KDA State Across Layers"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/mini_k3_all_layer_states.png",
    dpi=300
)

plt.show()
