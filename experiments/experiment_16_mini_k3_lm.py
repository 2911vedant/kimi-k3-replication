import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 16
# MINI K3 LANGUAGE MODEL
#
# KDA + SiTU-GLU + Token Embeddings + LM Head
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
# CONFIGURATION
# ============================================================

VOCAB_SIZE = 128

HIDDEN_DIM = 64

HEAD_DIM = 32

INTERMEDIATE_DIM = 128

NUM_BLOCKS = 2

SEQ_LEN = 64

BATCH_SIZE = 8

LEARNING_RATE = 0.003

TRAIN_STEPS = 500


# ============================================================
# SIMPLE CHARACTER TOKENIZER
# ============================================================

text = """
the quick brown fox jumps over the lazy dog.
the quick brown fox jumps over the lazy dog.
machine learning is a method of learning patterns from data.
large language models learn to predict the next token.
attention helps a model understand relationships between tokens.
kda provides a recurrent state for efficient sequence processing.
situation aware gated linear units improve nonlinear processing.
kimi is an example of a modern large language model architecture.
"""

# ------------------------------------------------------------
# Build vocabulary
# ------------------------------------------------------------

characters = sorted(
    set(text)
)

# Reserve IDs for unknown characters
stoi = {
    ch: i
    for i, ch in enumerate(characters)
}

itos = {
    i: ch
    for ch, i in stoi.items()
}


actual_vocab_size = len(stoi)

print()
print("Vocabulary size:", actual_vocab_size)


# ============================================================
# ENCODE TEXT
# ============================================================

data = torch.tensor(
    [
        stoi[ch]
        for ch in text
    ],
    dtype=torch.long
)


# ============================================================
# CREATE TRAINING BATCH
# ============================================================

def get_batch():

    starts = torch.randint(
        0,
        len(data) - SEQ_LEN - 1,
        (BATCH_SIZE,)
    )

    x = torch.stack(
        [
            data[i:i + SEQ_LEN]
            for i in starts
        ]
    )

    y = torch.stack(
        [
            data[i + 1:i + SEQ_LEN + 1]
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


        # ----------------------------------------------------
        # SiTU activation
        # ----------------------------------------------------

        gate_out = (
            self.beta1
            * torch.tanh(
                gate / self.beta1
            )
            * torch.sigmoid(gate)
        )


        # ----------------------------------------------------
        # Bounded up projection
        # ----------------------------------------------------

        up_out = (
            self.beta2
            * torch.tanh(
                up / self.beta2
            )
        )


        # ----------------------------------------------------
        # Gated multiplication
        # ----------------------------------------------------

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


        # ====================================================
        # PROCESS TOKENS
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
            # Delta
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


        # ----------------------------------------------------
        # Combine outputs
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # KDA
        # ----------------------------------------------------

        self.kda_norm = nn.LayerNorm(
            hidden_dim
        )

        self.kda = MiniKDA(
            hidden_dim,
            head_dim
        )


        # ----------------------------------------------------
        # SiTU-GLU
        # ----------------------------------------------------

        self.mlp_norm = nn.LayerNorm(
            hidden_dim
        )

        self.mlp = SiTUGLU(
            hidden_dim,
            intermediate_dim
        )


    def forward(self, x):


        # ----------------------------------------------------
        # KDA residual
        # ----------------------------------------------------

        x = x + self.kda(
            self.kda_norm(x)
        )


        # ----------------------------------------------------
        # SiTU residual
        # ----------------------------------------------------

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
        vocab_size,
        hidden_dim,
        head_dim,
        intermediate_dim,
        num_blocks
    ):

        super().__init__()


        # ----------------------------------------------------
        # Token embedding
        # ----------------------------------------------------

        self.token_embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )


        # ----------------------------------------------------
        # Transformer blocks
        # ----------------------------------------------------

        self.blocks = nn.ModuleList(
            [
                MiniK3Block(
                    hidden_dim,
                    head_dim,
                    intermediate_dim
                )

                for _ in range(num_blocks)
            ]
        )


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(
            hidden_dim
        )


        # ----------------------------------------------------
        # Language model head
        # ----------------------------------------------------

        self.lm_head = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False
        )


    def forward(
        self,
        tokens,
        targets=None
    ):

        # ----------------------------------------------------
        # Token → vectors
        # ----------------------------------------------------

        x = self.token_embedding(
            tokens
        )


        # ----------------------------------------------------
        # K3 blocks
        # ----------------------------------------------------

        for block in self.blocks:

            x = block(x)


        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        x = self.final_norm(
            x
        )


        # ----------------------------------------------------
        # Predict next token
        # ----------------------------------------------------

        logits = self.lm_head(
            x
        )


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
    vocab_size=actual_vocab_size,
    hidden_dim=HIDDEN_DIM,
    head_dim=HEAD_DIM,
    intermediate_dim=INTERMEDIATE_DIM,
    num_blocks=NUM_BLOCKS
).to(device)


# ============================================================
# MODEL INFORMATION
# ============================================================

parameters = sum(
    p.numel()
    for p in model.parameters()
)


print()
print("==========================================")
print("MINI K3 LANGUAGE MODEL")
print("==========================================")

print(
    "Vocabulary:",
    actual_vocab_size
)

print(
    "Hidden dimension:",
    HIDDEN_DIM
)

print(
    "KDA head dimension:",
    HEAD_DIM
)

print(
    "Intermediate dimension:",
    INTERMEDIATE_DIM
)

print(
    "Number of blocks:",
    NUM_BLOCKS
)

print(
    "Total parameters:",
    parameters
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

loss_history = []


print()
print("==========================================")
print("TRAINING")
print("==========================================")


model.train()


for step in range(
    TRAIN_STEPS
):

    x, y = get_batch()


    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    logits, loss = model(
        x,
        y
    )


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Print progress
    # --------------------------------------------------------

    if (
        step % 50 == 0
        or step == TRAIN_STEPS - 1
    ):

        print(
            f"Step {step:4d} | "
            f"Loss {loss.item():.4f}"
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
    "Mini K3 Language Model Training"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/mini_k3_training_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# GENERATION FUNCTION
# ============================================================

def generate(
    prompt,
    max_new_tokens=100
):

    model.eval()


    # --------------------------------------------------------
    # Encode prompt
    # --------------------------------------------------------

    tokens = [
        stoi.get(
            ch,
            0
        )

        for ch in prompt
    ]


    tokens = torch.tensor(
        tokens,
        dtype=torch.long,
        device=device
    ).unsqueeze(0)


    # --------------------------------------------------------
    # Generate
    # --------------------------------------------------------

    with torch.no_grad():

        for _ in range(
            max_new_tokens
        ):

            # Limit context

            context = tokens[
                :, -SEQ_LEN:
            ]


            logits, _ = model(
                context
            )


            # Last token

            next_logits = logits[
                :, -1, :
            ]


            # Temperature

            next_logits = (
                next_logits / 0.8
            )


            probabilities = F.softmax(
                next_logits,
                dim=-1
            )


            next_token = torch.multinomial(
                probabilities,
                num_samples=1
            )


            tokens = torch.cat(
                [
                    tokens,
                    next_token
                ],
                dim=1
            )


    # --------------------------------------------------------
    # Decode
    # --------------------------------------------------------

    generated = ""

    for token in tokens[0]:

        generated += itos[
            token.item()
        ]


    return generated


# ============================================================
# GENERATE TEXT
# ============================================================

print()
print("==========================================")
print("GENERATED TEXT")
print("==========================================")


prompt = "the "

generated_text = generate(
    prompt,
    max_new_tokens=100
)


print()
print(generated_text)


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("EXPERIMENT 16 COMPLETE")
print("==========================================")

print()
print("We now have:")

print("✓ Character tokenizer")

print("✓ Token embeddings")

print("✓ Mini KDA")

print("✓ SiTU-GLU")

print("✓ Multiple K3 blocks")

print("✓ Residual connections")

print("✓ Language-model head")

print("✓ Cross-entropy training")

print("✓ Text generation")

print()
print(
    "Loss graph saved to:"
)

print(
    "results/mini_k3_training_loss.png"
)
