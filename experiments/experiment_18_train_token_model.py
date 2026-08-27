import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 18
# MINI K3 + SUBWORD TOKENIZER
# ============================================================

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)


# ============================================================
# CONFIG
# ============================================================

HIDDEN_DIM = 64
HEAD_DIM = 32
INTERMEDIATE_DIM = 128
NUM_BLOCKS = 2

SEQ_LEN = 64
BATCH_SIZE = 8

TRAIN_STEPS = 1000
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

The purpose of this project is educational replication.
We are studying individual architectural ideas and
combining them into a small model that can run locally.
""" * 30


# ============================================================
# TOKENIZE
# ============================================================

tokens = tokenizer.encode(text)

data = torch.tensor(
    tokens,
    dtype=torch.long
)

vocab_size = tokenizer.n_vocab

print()
print("==========================================")
print("DATASET")
print("==========================================")

print("Tokens:", len(data))
print("Tokenizer vocabulary:", vocab_size)


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

        gate_fp32 = gate.float()
        up_fp32 = up.float()

        gate_out = (
            self.beta1
            * torch.tanh(
                gate_fp32 / self.beta1
            )
            * torch.sigmoid(
                gate_fp32
            )
        )

        up_out = (
            self.beta2
            * torch.tanh(
                up_fp32 / self.beta2
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

            delta = (
                v_t - prediction
            )

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
# MINI K3 BLOCK
# ============================================================

class MiniK3Block(nn.Module):

    def __init__(
        self,
        hidden_dim,
        head_dim,
        intermediate_dim
    ):

        super().__init__()

        self.kda_norm = nn.LayerNorm(
            hidden_dim
        )

        self.kda = MiniKDA(
            hidden_dim,
            head_dim
        )

        self.mlp_norm = nn.LayerNorm(
            hidden_dim
        )

        self.mlp = SiTUGLU(
            hidden_dim,
            intermediate_dim
        )

    def forward(self, x):

        x = x + self.kda(
            self.kda_norm(x)
        )

        x = x + self.mlp(
            self.mlp_norm(x)
        )

        return x


# ============================================================
# MINI K3 LANGUAGE MODEL
# ============================================================

class MiniK3LM(nn.Module):

    def __init__(
        self,
        vocab_size
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            HIDDEN_DIM
        )

        self.blocks = nn.ModuleList(
            [
                MiniK3Block(
                    HIDDEN_DIM,
                    HEAD_DIM,
                    INTERMEDIATE_DIM
                )

                for _ in range(
                    NUM_BLOCKS
                )
            ]
        )

        self.final_norm = nn.LayerNorm(
            HIDDEN_DIM
        )

        self.lm_head = nn.Linear(
            HIDDEN_DIM,
            vocab_size,
            bias=False
        )

    def forward(
        self,
        tokens,
        targets=None
    ):

        x = self.embedding(
            tokens
        )

        for block in self.blocks:

            x = block(x)

        x = self.final_norm(x)

        logits = self.lm_head(x)

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
# CREATE MODEL
# ============================================================

model = MiniK3LM(
    vocab_size
).to(device)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
)


print()
print("==========================================")
print("MODEL")
print("==========================================")

print(
    "Parameters:",
    parameter_count
)

print(
    "Blocks:",
    NUM_BLOCKS
)

print(
    "Hidden dimension:",
    HIDDEN_DIM
)

print(
    "Sequence length:",
    SEQ_LEN
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAIN
# ============================================================

loss_history = []

model.train()

print()
print("==========================================")
print("TRAINING")
print("==========================================")


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

    loss_history.append(
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


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(10, 6)
)

plt.plot(
    loss_history
)

plt.xlabel(
    "Training step"
)

plt.ylabel(
    "Cross-entropy loss"
)

plt.title(
    "Mini K3 — Subword Training"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/mini_k3_subword_training.png",
    dpi=300
)

plt.show()


# ============================================================
# GENERATION
# ============================================================

def generate(
    prompt,
    max_new_tokens=100,
    temperature=0.8
):

    model.eval()

    prompt_tokens = tokenizer.encode(
        prompt
    )

    tokens = torch.tensor(
        prompt_tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            context = tokens[
                :, -SEQ_LEN:
            ]

            logits, _ = model(
                context
            )

            next_logits = logits[
                :, -1, :
            ]

            next_logits = (
                next_logits
                / temperature
            )

            probabilities = F.softmax(
                next_logits,
                dim=-1
            )

            next_token = torch.multinomial(
                probabilities,
                1
            )

            tokens = torch.cat(
                [
                    tokens,
                    next_token
                ],
                dim=1
            )

    return tokenizer.decode(
        tokens[0].tolist()
    )


# ============================================================
# TEST GENERATION
# ============================================================

print()
print("==========================================")
print("GENERATED TEXT")
print("==========================================")

for prompt in [
    "Kimi K3",
    "SiTU-GLU",
    "Kimi Delta Attention"
]:

    print()
    print("PROMPT:", prompt)
    print("------------------------------------------")

    result = generate(
        prompt,
        max_new_tokens=80
    )

    print(result)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    model.state_dict(),
    "results/mini_k3_subword_model.pt"
)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("EXPERIMENT 18 COMPLETE")
print("==========================================")

print()
print("Saved:")

print(
    "results/mini_k3_subword_training.png"
)

print(
    "results/mini_k3_subword_model.pt"
)
