import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# Mini KDA Head
# ============================================================

class MiniKDA(nn.Module):

    def __init__(self, input_dim, head_dim, kernel_size=3):
        super().__init__()

        self.input_dim = input_dim
        self.head_dim = head_dim

        # ----------------------------------------------------
        # Q / K / V projections
        # ----------------------------------------------------

        self.q_proj = nn.Linear(input_dim, head_dim)
        self.k_proj = nn.Linear(input_dim, head_dim)
        self.v_proj = nn.Linear(input_dim, head_dim)

        # ----------------------------------------------------
        # Short convolution
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Alpha retention gate
        # ----------------------------------------------------

        self.alpha_proj = nn.Linear(
            input_dim,
            head_dim
        )

        # ----------------------------------------------------
        # Beta write gate
        # ----------------------------------------------------

        self.beta_proj = nn.Linear(
            input_dim,
            1
        )


    # ========================================================
    # ShortConv
    # ========================================================

    def short_conv(self, x, conv):

        # [batch, sequence, dimension]
        x = x.transpose(1, 2)

        x = conv(x)

        # Remove future positions introduced by padding
        trim = conv.kernel_size[0] - 1

        if trim > 0:
            x = x[:, :, :-trim]

        x = x.transpose(1, 2)

        return x


    # ========================================================
    # Forward
    # ========================================================

    def forward(self, x):

        batch_size, sequence_length, _ = x.shape

        # ----------------------------------------------------
        # Q / K / V
        # ----------------------------------------------------

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # ----------------------------------------------------
        # ShortConv
        # ----------------------------------------------------

        q = self.short_conv(q, self.q_conv)
        k = self.short_conv(k, self.k_conv)
        v = self.short_conv(v, self.v_conv)

        # ----------------------------------------------------
        # Swish
        # ----------------------------------------------------

        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        # ----------------------------------------------------
        # L2 normalize Q and K
        # ----------------------------------------------------

        q = F.normalize(q, p=2, dim=-1)
        k = F.normalize(k, p=2, dim=-1)

        # ----------------------------------------------------
        # Alpha = retention gate
        # ----------------------------------------------------

        alpha = torch.sigmoid(
            self.alpha_proj(x)
        )

        # ----------------------------------------------------
        # Beta = write strength
        # ----------------------------------------------------

        beta = torch.sigmoid(
            self.beta_proj(x)
        )

        # ----------------------------------------------------
        # Recurrent KDA state
        # ----------------------------------------------------

        state = torch.zeros(
            batch_size,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        state_norms = []

        # ----------------------------------------------------
        # Process tokens sequentially
        # ----------------------------------------------------

        for t in range(sequence_length):

            qt = q[:, t]
            kt = k[:, t]
            vt = v[:, t]

            alphat = alpha[:, t]
            betat = beta[:, t]

            batch_states = []

            batch_outputs = []

            for b in range(batch_size):

                S = state[b]

                q_vec = qt[b]
                k_vec = kt[b]
                v_vec = vt[b]

                alpha_vec = alphat[b]
                beta_scalar = betat[b, 0]

                # ------------------------------------------
                # Channel-wise retention
                # ------------------------------------------

                retained = (
                    alpha_vec.unsqueeze(1) * S
                )

                # ------------------------------------------
                # Delta correction
                # ------------------------------------------

                I = torch.eye(
                    self.head_dim,
                    device=x.device,
                    dtype=x.dtype
                )

                correction_matrix = (
                    I
                    - beta_scalar
                    * torch.outer(k_vec, k_vec)
                )

                corrected = (
                    correction_matrix @ retained
                )

                # ------------------------------------------
                # Write new information
                # ------------------------------------------

                new_information = (
                    beta_scalar
                    * torch.outer(k_vec, v_vec)
                )

                # ------------------------------------------
                # New state
                # ------------------------------------------

                S_new = (
                    corrected
                    + new_information
                )

                # ------------------------------------------
                # Read using query
                # ------------------------------------------

                output = S_new.T @ q_vec

                batch_states.append(S_new)
                batch_outputs.append(output)

            state = torch.stack(batch_states)
            output = torch.stack(batch_outputs)

            outputs.append(output)
            state_norms.append(
                torch.norm(state).item()
            )

        # [sequence, batch, dimension]
        outputs = torch.stack(outputs)

        # [batch, sequence, dimension]
        outputs = outputs.transpose(0, 1)

        return outputs, state_norms, alpha, beta


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


# Small model
batch_size = 1
sequence_length = 32
input_dim = 32
head_dim = 16


# ============================================================
# Random input
# ============================================================

x = torch.randn(
    batch_size,
    sequence_length,
    input_dim,
    device=device
)


# ============================================================
# Create KDA
# ============================================================

model = MiniKDA(
    input_dim=input_dim,
    head_dim=head_dim
).to(device)


# ============================================================
# Forward pass
# ============================================================

output, state_norms, alpha, beta = model(x)


# ============================================================
# Print information
# ============================================================

print()
print("Input shape :", x.shape)
print("Output shape:", output.shape)

print()
print(
    "Average alpha:",
    alpha.mean().item()
)

print(
    "Average beta:",
    beta.mean().item()
)

print(
    "Final state norm:",
    state_norms[-1]
)


# ============================================================
# Plot state evolution
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
    "Mini KDA State Evolution"
)

plt.grid(True)

plt.savefig(
    "results/mini_kda_state.png",
    dpi=300
)

plt.show()


# ============================================================
# Plot alpha and beta
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    alpha[0].mean(dim=-1).detach().cpu().numpy(),
    label="Average α"
)

plt.plot(
    beta[0, :, 0].detach().cpu().numpy(),
    label="β"
)

plt.xlabel("Token")
plt.ylabel("Gate value")

plt.title(
    "Mini KDA Retention and Write Gates"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/mini_kda_gates.png",
    dpi=300
)

plt.show()
