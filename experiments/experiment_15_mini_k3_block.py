import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 15
# MINI K3 BLOCK
#
# KDA + SiTU-GLU
# ============================================================

torch.manual_seed(42)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# SiTU-GLU
# ============================================================

class SiTUGLU(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        beta1=4.0,
        beta2=25.0
    ):
        super().__init__()

        self.beta1 = beta1
        self.beta2 = beta2

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

        # ----------------------------------------------------
        # SiTU gate
        # ----------------------------------------------------

        gate_out = (
            self.beta1
            * torch.tanh(
                gate / self.beta1
            )
            * torch.sigmoid(gate)
        )

        # ----------------------------------------------------
        # Bounded up branch
        # ----------------------------------------------------

        up_out = (
            self.beta2
            * torch.tanh(
                up / self.beta2
            )
        )

        # ----------------------------------------------------
        # GLU multiplication
        # ----------------------------------------------------

        output = gate_out * up_out

        output = self.down_proj(
            output.to(x.dtype)
        )

        return output


# ============================================================
# MINI KDA
# ============================================================

class MiniKDA(nn.Module):

    def __init__(
        self,
        hidden_dim=32,
        head_dim=16
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

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

        self.o_proj = nn.Linear(
            head_dim,
            hidden_dim,
            bias=False
        )

        # Learnable decay
        self.decay = nn.Parameter(
            torch.zeros(head_dim)
        )


    def forward(self, x):

        batch_size = x.shape[0]

        sequence_length = x.shape[1]


        # ----------------------------------------------------
        # Q K V
        # ----------------------------------------------------

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)


        # ----------------------------------------------------
        # Normalize
        # ----------------------------------------------------

        q = F.normalize(
            q,
            dim=-1
        )

        k = F.normalize(
            k,
            dim=-1
        )


        # ----------------------------------------------------
        # Decay
        # ----------------------------------------------------

        decay = torch.sigmoid(
            self.decay
        )


        # ----------------------------------------------------
        # Recurrent state
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


        # ====================================================
        # RECURRENT PROCESSING
        # ====================================================

        for t in range(
            sequence_length
        ):

            q_t = q[:, t]

            k_t = k[:, t]

            v_t = v[:, t]


            # ------------------------------------------------
            # State prediction
            # ------------------------------------------------

            prediction = torch.bmm(
                k_t.unsqueeze(1),
                state
            ).squeeze(1)


            # ------------------------------------------------
            # Delta correction
            # ------------------------------------------------

            delta = (
                v_t
                - prediction
            )


            # ------------------------------------------------
            # Decay old state
            # ------------------------------------------------

            state = state * decay.view(
                1,
                -1,
                1
            )


            # ------------------------------------------------
            # Write correction
            # ------------------------------------------------

            state = state + torch.bmm(
                k_t.unsqueeze(2),
                delta.unsqueeze(1)
            )


            # ------------------------------------------------
            # Read state
            # ------------------------------------------------

            output_t = torch.bmm(
                q_t.unsqueeze(1),
                state
            ).squeeze(1)


            outputs.append(
                output_t
            )


            state_norms.append(
                state.norm(
                    dim=(1, 2)
                ).mean().item()
            )


        output = torch.stack(
            outputs,
            dim=1
        )


        output = self.o_proj(
            output
        )


        return output, state_norms


# ============================================================
# MINI K3 BLOCK
# ============================================================

class MiniK3Block(nn.Module):

    def __init__(
        self,
        hidden_dim=32,
        head_dim=16,
        intermediate_dim=64
    ):
        super().__init__()


        # ----------------------------------------------------
        # KDA
        # ----------------------------------------------------

        self.kda_norm = nn.LayerNorm(
            hidden_dim
        )

        self.kda = MiniKDA(
            hidden_dim=hidden_dim,
            head_dim=head_dim
        )


        # ----------------------------------------------------
        # SiTU-GLU
        # ----------------------------------------------------

        self.mlp_norm = nn.LayerNorm(
            hidden_dim
        )

        self.mlp = SiTUGLU(
            hidden_dim=hidden_dim,
            intermediate_dim=intermediate_dim
        )


    def forward(self, x):


        # ====================================================
        # KDA
        # ====================================================

        kda_input = self.kda_norm(
            x
        )

        kda_output, state_norms = self.kda(
            kda_input
        )

        x = x + kda_output


        # ====================================================
        # SiTU-GLU
        # ====================================================

        mlp_input = self.mlp_norm(
            x
        )

        mlp_output = self.mlp(
            mlp_input
        )

        x = x + mlp_output


        return x, state_norms


# ============================================================
# CREATE BLOCK
# ============================================================

hidden_dim = 32

head_dim = 16

intermediate_dim = 64

sequence_length = 100

batch_size = 2


model = MiniK3Block(
    hidden_dim=hidden_dim,
    head_dim=head_dim,
    intermediate_dim=intermediate_dim
).to(device)


# ============================================================
# MODEL INFORMATION
# ============================================================

parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print("==========================================")
print("MINI K3 BLOCK")
print("==========================================")

print(
    "Hidden dimension :",
    hidden_dim
)

print(
    "KDA head dimension:",
    head_dim
)

print(
    "MLP dimension    :",
    intermediate_dim
)

print(
    "Sequence length  :",
    sequence_length
)

print(
    "Parameters       :",
    parameter_count
)


# ============================================================
# RANDOM INPUT
# ============================================================

x = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    device=device
)


print()
print(
    "Input shape:",
    tuple(x.shape)
)


# ============================================================
# FORWARD PASS
# ============================================================

output, state_norms = model(
    x
)


print(
    "Output shape:",
    tuple(output.shape)
)


# ============================================================
# CHECK RESIDUAL CONNECTION
# ============================================================

difference = (
    output - x
).abs().mean().item()


print(
    "Mean input/output difference:",
    difference
)


# ============================================================
# CHECK GRADIENTS
# ============================================================

loss = output.mean()

loss.backward()


gradient_count = 0

for name, parameter in model.named_parameters():

    if parameter.grad is not None:

        gradient_count += 1


print(
    "Parameters receiving gradients:",
    gradient_count
)


# ============================================================
# STATE GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    range(
        1,
        sequence_length + 1
    ),
    state_norms
)

plt.xlabel(
    "Token"
)

plt.ylabel(
    "KDA state norm"
)

plt.title(
    "Mini K3 Block — KDA State Evolution"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/mini_k3_block_state.png",
    dpi=300
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

print()
print("==========================================")
print("EXPERIMENT 15 COMPLETE")
print("==========================================")

print()
print("Our block now contains:")

print("1. LayerNorm")

print("2. KDA")

print("3. Residual connection")

print("4. LayerNorm")

print("5. SiTU-GLU")

print("6. Residual connection")

print()
print(
    "Graph saved:"
)

print(
    "results/mini_k3_block_state.png"
)
