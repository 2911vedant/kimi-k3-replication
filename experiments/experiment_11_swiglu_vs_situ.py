import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# DEVICE
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# SwiGLU
# ============================================================

class SwiGLU(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim
    ):
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

    def forward(self, x):

        gate = self.gate_proj(x)

        up = self.up_proj(x)

        # SiLU gate
        gate = F.silu(gate)

        # GLU multiplication
        output = gate * up

        # Down projection
        output = self.down_proj(output)

        return output


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

        # Compute activation in FP32
        gate_fp32 = gate.float()
        up_fp32 = up.float()

        # ----------------------------------------------------
        # SiTU gate
        # ----------------------------------------------------

        gate_out = (
            self.beta
            * torch.tanh(
                gate_fp32 / self.beta
            )
            * torch.sigmoid(
                gate_fp32
            )
        )

        # ----------------------------------------------------
        # Bounded linear branch
        # ----------------------------------------------------

        up_out = (
            self.linear_beta
            * torch.tanh(
                up_fp32 / self.linear_beta
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
# SIMPLE TRANSFORMER BLOCK
# ============================================================

class MiniBlock(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        activation
    ):
        super().__init__()

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        if activation == "swiglu":

            self.mlp = SwiGLU(
                hidden_dim,
                intermediate_dim
            )

        elif activation == "situ":

            self.mlp = SiTUGLU(
                hidden_dim,
                intermediate_dim
            )

        else:

            raise ValueError(
                "activation must be swiglu or situ"
            )

    def forward(self, x):

        x_norm = self.norm(x)

        return x + self.mlp(x_norm)


# ============================================================
# LANGUAGE MODEL
# ============================================================

class MiniLanguageModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=3,
        activation="swiglu"
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

        self.layers = nn.ModuleList(
            [
                MiniBlock(
                    hidden_dim,
                    intermediate_dim,
                    activation
                )
                for _ in range(num_layers)
            ]
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

        for layer in self.layers:

            x = layer(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

        return logits


# ============================================================
# DATASET
# ============================================================

text = """
kimi k3 is a language model.
kda is a linear attention mechanism.
situ glu is a gated activation.
attention residuals combine layer representations.
mixture of experts routes tokens to experts.
kimi k3 combines efficient architectures.
""" * 30


# ============================================================
# CHARACTER VOCABULARY
# ============================================================

chars = sorted(
    list(set(text))
)

vocab_size = len(chars)

stoi = {
    ch: i
    for i, ch in enumerate(chars)
}

itos = {
    i: ch
    for i, ch in enumerate(chars)
}


encoded = torch.tensor(
    [
        stoi[c]
        for c in text
    ],
    dtype=torch.long
)


print()
print("Vocabulary size:", vocab_size)
print("Dataset size   :", len(encoded))


# ============================================================
# TRAINING SETTINGS
# ============================================================

sequence_length = 32

batch_size = 8

training_steps = 500

learning_rate = 3e-4


# ============================================================
# BATCH GENERATOR
# ============================================================

def get_batch():

    starts = torch.randint(
        0,
        len(encoded)
        - sequence_length
        - 1,
        (batch_size,)
    )

    x = torch.stack(
        [
            encoded[
                i:
                i + sequence_length
            ]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            encoded[
                i + 1:
                i + sequence_length + 1
            ]
            for i in starts
        ]
    )

    return (
        x.to(device),
        y.to(device)
    )


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(
    activation
):

    print()
    print(
        "===================================="
    )

    print(
        f"Training {activation.upper()}"
    )

    print(
        "===================================="
    )

    # Same random seed for fair comparison
    torch.manual_seed(42)

    model = MiniLanguageModel(
        vocab_size=vocab_size,
        hidden_dim=32,
        intermediate_dim=64,
        num_layers=3,
        activation=activation
    ).to(device)

    parameter_count = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        parameter_count
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    losses = []

    model.train()

    for step in range(
        training_steps
    ):

        x, y = get_batch()

        logits = model(x)

        loss = F.cross_entropy(
            logits.reshape(
                -1,
                vocab_size
            ),
            y.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            1.0
        )

        optimizer.step()

        losses.append(
            loss.item()
        )

        if (
            step == 0
            or (step + 1) % 50 == 0
        ):

            print(
                f"Step {step + 1:4d} | "
                f"Loss: {loss.item():.4f}"
            )

    return model, losses


# ============================================================
# TRAIN SwiGLU
# ============================================================

swiglu_model, swiglu_losses = train_model(
    "swiglu"
)


# ============================================================
# TRAIN SiTU-GLU
# ============================================================

situ_model, situ_losses = train_model(
    "situ"
)


# ============================================================
# FINAL LOSS
# ============================================================

print()
print(
    "===================================="
)

print(
    "FINAL RESULTS"
)

print(
    "===================================="
)

print(
    "SwiGLU final loss:",
    swiglu_losses[-1]
)

print(
    "SiTU-GLU final loss:",
    situ_losses[-1]
)


# ============================================================
# PLOT BOTH LOSSES
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    swiglu_losses,
    label="SwiGLU"
)

plt.plot(
    situ_losses,
    label="SiTU-GLU"
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Cross-entropy loss"
)

plt.title(
    "SwiGLU vs SiTU-GLU"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/swiglu_vs_situ_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# SMOOTHED LOSS
# ============================================================

def moving_average(
    values,
    window=25
):

    result = []

    for i in range(
        len(values)
    ):

        start = max(
            0,
            i - window + 1
        )

        result.append(
            sum(
                values[start:i + 1]
            )
            / (
                i - start + 1
            )
        )

    return result


swiglu_smooth = moving_average(
    swiglu_losses
)

situ_smooth = moving_average(
    situ_losses
)


plt.figure(figsize=(10, 6))

plt.plot(
    swiglu_smooth,
    label="SwiGLU"
)

plt.plot(
    situ_smooth,
    label="SiTU-GLU"
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Smoothed loss"
)

plt.title(
    "SwiGLU vs SiTU-GLU — Smoothed"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/swiglu_vs_situ_smoothed.png",
    dpi=300
)

plt.show()


print()
print(
    "===================================="
)

print(
    "EXPERIMENT 11 COMPLETE"
)

print(
    "===================================="
)

print(
    "Graphs saved:"
)

print(
    "results/swiglu_vs_situ_loss.png"
)

print(
    "results/swiglu_vs_situ_smoothed.png"
)
