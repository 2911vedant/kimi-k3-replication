import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 19
# MINI-K3 vs STANDARD CAUSAL ATTENTION
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIGURATION
# ============================================================

HIDDEN_DIM = 64
HEAD_DIM = 32
INTERMEDIATE_DIM = 128

NUM_BLOCKS = 2

SEQ_LEN = 64
BATCH_SIZE = 8

TRAIN_STEPS = 500
LEARNING_RATE = 0.001


# ============================================================
# TOKENIZER
# ============================================================

tokenizer = tiktoken.get_encoding(
    "cl100k_base"
)


# ============================================================
# DATASET
# ============================================================

text = """
Kimi K3 is a large language model architecture.
Large language models process sequences of tokens.
A language model learns to predict the next token.

Kimi K3 introduces several important architectural ideas.
Kimi Delta Attention provides an efficient recurrent
attention mechanism for processing long sequences.

SiTU-GLU is a gated activation function.
The activation uses smooth bounding functions to control
the magnitude of the activation.

SwiGLU is another gated linear unit commonly used in
modern transformer language models.

The goal of this project is to build a small educational
model inspired by the ideas used in Kimi K3.

The model contains KDA and SiTU-GLU components.
The model is trained using next token prediction.

Machine learning models learn patterns from data.
Language models learn relationships between tokens.
Training reduces the prediction error over time.

Attention mechanisms allow models to combine information
from different parts of a sequence.

Efficient architectures are important because modern
language models can contain billions of parameters.

Kimi Delta Attention maintains a recurrent state while
processing tokens sequentially.

SiTU-GLU uses bounded nonlinear transformations.

The Mini K3 model combines these ideas into a small
trainable language model for experimentation.
""" * 30


tokens = tokenizer.encode(text)

data = torch.tensor(
    tokens,
    dtype=torch.long
)

vocab_size = tokenizer.n_vocab


# ============================================================
# BATCH GENERATOR
# ============================================================

def get_batch():

    starts = torch.randint(
        0,
        len(data) - SEQ_LEN - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            data[
                i:i + SEQ_LEN
            ]

            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[
                i + 1:i + SEQ_LEN + 1
            ]

            for i in starts
        ]
    )

    return (
        x.to(device),
        y.to(device)
    )


# ============================================================
# SiTU-GLU
# ============================================================

class SiTUGLU(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim
    ):

        super().__init__()

        self.beta1 = 4.0
        self.beta2 = 25.0

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

        gate_out = (
            self.beta1
            * torch.tanh(
                gate / self.beta1
            )
            * torch.sigmoid(gate)
        )

        up_out = (
            self.beta2
            * torch.tanh(
                up / self.beta2
            )
        )

        output = gate_out * up_out

        return self.down_proj(
            output.to(x.dtype)
        )


# ============================================================
# MINI KDA
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

        self.o_proj = nn.Linear(
            head_dim,
            hidden_dim,
            bias=False
        )

        self.decay = nn.Parameter(
            torch.zeros(head_dim)
        )

    def forward(self, x):

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        q = F.normalize(
            self.q_proj(x),
            dim=-1
        )

        k = F.normalize(
            self.k_proj(x),
            dim=-1
        )

        v = self.v_proj(x)

        decay = torch.sigmoid(
            self.decay
        )

        state = torch.zeros(
            batch_size,
            self.head_dim,
            self.head_dim,
            device=x.device,
            dtype=x.dtype
        )

        outputs = []

        for t in range(
            sequence_length
        ):

            q_t = q[:, t]
            k_t = k[:, t]
            v_t = v[:, t]

            prediction = torch.bmm(
                k_t.unsqueeze(1),
                state
            ).squeeze(1)

            delta = v_t - prediction

            state = state * decay.view(
                1,
                -1,
                1
            )

            state = state + torch.bmm(
                k_t.unsqueeze(2),
                delta.unsqueeze(1)
            )

            output_t = torch.bmm(
                q_t.unsqueeze(1),
                state
            ).squeeze(1)

            outputs.append(
                output_t
            )

        output = torch.stack(
            outputs,
            dim=1
        )

        return self.o_proj(
            output
        )


# ============================================================
# STANDARD CAUSAL SELF-ATTENTION
# ============================================================

class StandardAttention(nn.Module):

    def __init__(
        self,
        hidden_dim,
        head_dim
    ):

        super().__init__()

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

        self.head_dim = head_dim


    def forward(self, x):

        q = self.q_proj(x)

        k = self.k_proj(x)

        v = self.v_proj(x)

        scores = torch.matmul(
            q,
            k.transpose(
                -2,
                -1
            )
        )

        scores = scores / (
            self.head_dim ** 0.5
        )


        # ----------------------------------------------------
        # Causal mask
        # ----------------------------------------------------

        sequence_length = x.shape[1]

        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=x.device,
                dtype=torch.bool
            ),
            diagonal=1
        )


        scores = scores.masked_fill(
            mask,
            float("-inf")
        )


        attention = F.softmax(
            scores,
            dim=-1
        )


        output = torch.matmul(
            attention,
            v
        )


        return self.o_proj(
            output
        )


# ============================================================
# MINI-K3 BLOCK
# ============================================================

class MiniK3Block(nn.Module):

    def __init__(self):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.attention = MiniKDA(
            HIDDEN_DIM,
            HEAD_DIM
        )

        self.norm2 = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.mlp = SiTUGLU(
            HIDDEN_DIM,
            INTERMEDIATE_DIM
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# BASELINE BLOCK
# ============================================================

class BaselineBlock(nn.Module):

    def __init__(self):

        super().__init__()

        self.norm1 = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.attention = StandardAttention(
            HIDDEN_DIM,
            HEAD_DIM
        )

        self.norm2 = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.mlp = SiTUGLU(
            HIDDEN_DIM,
            INTERMEDIATE_DIM
        )

    def forward(self, x):

        x = x + self.attention(
            self.norm1(x)
        )

        x = x + self.mlp(
            self.norm2(x)
        )

        return x


# ============================================================
# MINI-K3 MODEL
# ============================================================

class MiniK3(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            HIDDEN_DIM
        )

        self.blocks = nn.ModuleList(
            [
                MiniK3Block()
                for _ in range(NUM_BLOCKS)
            ]
        )

        self.norm = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.head = nn.Linear(
            HIDDEN_DIM,
            vocab_size,
            bias=False
        )

    def forward(
        self,
        x,
        targets=None
    ):

        x = self.embedding(x)

        for block in self.blocks:

            x = block(x)

        x = self.norm(x)

        logits = self.head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

        return logits, loss


# ============================================================
# BASELINE MODEL
# ============================================================

class BaselineTransformer(nn.Module):

    def __init__(self):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            HIDDEN_DIM
        )

        self.blocks = nn.ModuleList(
            [
                BaselineBlock()
                for _ in range(NUM_BLOCKS)
            ]
        )

        self.norm = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.head = nn.Linear(
            HIDDEN_DIM,
            vocab_size,
            bias=False
        )

    def forward(
        self,
        x,
        targets=None
    ):

        x = self.embedding(x)

        for block in self.blocks:

            x = block(x)

        x = self.norm(x)

        logits = self.head(x)

        loss = None

        if targets is not None:

            loss = F.cross_entropy(
                logits.reshape(
                    -1,
                    logits.size(-1)
                ),
                targets.reshape(-1)
            )

        return logits, loss


# ============================================================
# TRAIN FUNCTION
# ============================================================

def train_model(
    model,
    name
):

    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    losses = []

    model.train()

    print()
    print("==========================================")
    print("TRAINING:", name)
    print("==========================================")

    parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Parameters:",
        parameters
    )

    for step in range(
        TRAIN_STEPS
    ):

        x, y = get_batch()

        logits, loss = model(
            x,
            y
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
# CREATE MODELS
# ============================================================

k3_model = MiniK3()

baseline_model = BaselineTransformer()


# ============================================================
# TRAIN
# ============================================================

k3_model, k3_losses = train_model(
    k3_model,
    "Mini-K3 / KDA"
)


baseline_model, baseline_losses = train_model(
    baseline_model,
    "Baseline / Standard Attention"
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("==========================================")
print("FINAL COMPARISON")
print("==========================================")

print(
    "Mini-K3 final loss:",
    k3_losses[-1]
)

print(
    "Baseline final loss:",
    baseline_losses[-1]
)


# ============================================================
# LOSS CURVE
# ============================================================

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    k3_losses,
    label="Mini-K3 / KDA"
)

plt.plot(
    baseline_losses,
    label="Baseline / Standard Attention"
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Cross-entropy loss"
)

plt.title(
    "Mini-K3 vs Standard Attention"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/k3_vs_baseline.png",
    dpi=300
)

plt.show()


# ============================================================
# SMOOTHED CURVE
# ============================================================

def moving_average(
    values,
    window=30
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


k3_smooth = moving_average(
    k3_losses
)

baseline_smooth = moving_average(
    baseline_losses
)


plt.figure(
    figsize=(11, 6)
)

plt.plot(
    k3_smooth,
    label="Mini-K3 / KDA"
)

plt.plot(
    baseline_smooth,
    label="Baseline / Standard Attention"
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Smoothed loss"
)

plt.title(
    "Mini-K3 vs Baseline — Smoothed"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/k3_vs_baseline_smoothed.png",
    dpi=300
)

plt.show()


# ============================================================
# SAVE RESULTS
# ============================================================

torch.save(
    k3_model.state_dict(),
    "results/mini_k3_model.pt"
)

torch.save(
    baseline_model.state_dict(),
    "results/baseline_model.pt"
)


print()
print("==========================================")
print("EXPERIMENT 19 COMPLETE")
print("==========================================")

print()
print("Graphs:")

print(
    "results/k3_vs_baseline.png"
)

print(
    "results/k3_vs_baseline_smoothed.png"
)

print()
print("Models:")

print(
    "results/mini_k3_model.pt"
)

print(
    "results/baseline_model.pt"
)
