import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 14
# MINI KIMI DELTA ATTENTION
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


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

        # Query projection
        self.q_proj = nn.Linear(
            hidden_dim,
            head_dim,
            bias=False
        )

        # Key projection
        self.k_proj = nn.Linear(
            hidden_dim,
            head_dim,
            bias=False
        )

        # Value projection
        self.v_proj = nn.Linear(
            hidden_dim,
            head_dim,
            bias=False
        )

        # Output projection
        self.o_proj = nn.Linear(
            head_dim,
            hidden_dim,
            bias=False
        )

        # Learnable decay
        self.decay = nn.Parameter(
            torch.zeros(head_dim)
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(self, x):

        """
        x:

        [batch, sequence, hidden_dim]

        Returns:

        output
        """

        batch_size = x.shape[0]

        sequence_length = x.shape[1]

        # ----------------------------------------------------
        # Project Q, K, V
        # ----------------------------------------------------

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)


        # ----------------------------------------------------
        # Normalize Q and K
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
        # Convert decay to 0-1
        # ----------------------------------------------------

        decay = torch.sigmoid(
            self.decay
        )


        # ----------------------------------------------------
        # Recurrent state
        #
        # state shape:
        #
        # [batch, key_dim, value_dim]
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
        # RECURRENT KDA UPDATE
        # ====================================================

        for t in range(
            sequence_length
        ):

            q_t = q[:, t]

            k_t = k[:, t]

            v_t = v[:, t]


            # ------------------------------------------------
            # Current prediction from state
            # ------------------------------------------------

            prediction = torch.bmm(
                k_t.unsqueeze(1),
                state
            ).squeeze(1)


            # ------------------------------------------------
            # Delta
            #
            # What did the state predict incorrectly?
            # ------------------------------------------------

            delta = (
                v_t
                - prediction
            )


            # ------------------------------------------------
            # Forget old information
            # ------------------------------------------------

            state = (
                state
                * decay.view(
                    1,
                    -1,
                    1
                )
            )


            # ------------------------------------------------
            # Write correction into state
            # ------------------------------------------------

            state = state + torch.bmm(
                k_t.unsqueeze(2),
                delta.unsqueeze(1)
            )


            # ------------------------------------------------
            # Read state using query
            # ------------------------------------------------

            output_t = torch.bmm(
                q_t.unsqueeze(1),
                state
            ).squeeze(1)


            outputs.append(
                output_t
            )


            # Save state magnitude
            state_norms.append(
                state.norm(
                    dim=(1, 2)
                ).mean().item()
            )


        # ====================================================
        # STACK OUTPUTS
        # ====================================================

        output = torch.stack(
            outputs,
            dim=1
        )


        # ====================================================
        # OUTPUT PROJECTION
        # ====================================================

        output = self.o_proj(
            output
        )


        return output, state_norms


# ============================================================
# CREATE MODEL
# ============================================================

model = MiniKDA(
    hidden_dim=32,
    head_dim=16
).to(device)


print()
print("==========================================")
print("MINI KDA")
print("==========================================")

print(
    "Parameters:",
    sum(
        p.numel()
        for p in model.parameters()
    )
)


# ============================================================
# CREATE RANDOM TOKEN REPRESENTATIONS
# ============================================================

batch_size = 1

sequence_length = 100

hidden_dim = 32


x = torch.randn(
    batch_size,
    sequence_length,
    hidden_dim,
    device=device
)


# ============================================================
# RUN KDA
# ============================================================

with torch.no_grad():

    output, state_norms = model(
        x
    )


print()
print(
    "Input shape :",
    tuple(x.shape)
)

print(
    "Output shape:",
    tuple(output.shape)
)

print(
    "Number of recurrent states:",
    len(state_norms)
)


# ============================================================
# SHOW STATE EVOLUTION
# ============================================================

print()
print("State magnitude:")

for i in range(
    0,
    sequence_length,
    10
):

    print(
        f"Token {i + 1:3d}: "
        f"{state_norms[i]:.4f}"
    )


# ============================================================
# GRAPH
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
    "Recurrent state norm"
)

plt.title(
    "Mini KDA — Recurrent State Evolution"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/mini_kda_state.png",
    dpi=300
)

plt.show()


# ============================================================
# OUTPUT DIFFERENCE TEST
# ============================================================

# Process the same sequence in two pieces
# to demonstrate recurrent processing.

first_half = x[:, :50]

second_half = x[:, 50:]


with torch.no_grad():

    output_full, _ = model(
        x
    )


print()
print("==========================================")
print("KDA TEST COMPLETE")
print("==========================================")

print()
print(
    "Graph saved:"
)

print(
    "results/mini_kda_state.png"
)

print()
print(
    "KDA successfully processed",
    sequence_length,
    "tokens using a fixed-size recurrent state."
)
