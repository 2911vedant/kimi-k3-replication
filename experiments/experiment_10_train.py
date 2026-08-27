import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# ============================================================
# SiTU-GLU
# ============================================================

class SiTUGLU(nn.Module):

    def __init__(self, hidden_dim, intermediate_dim):
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

        self.beta = 4.0
        self.linear_beta = 25.0

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

        return self.down_proj(
            output.to(x.dtype)
        )


# ============================================================
# Mini KDA
# ============================================================

class MiniKDA(nn.Module):

    def __init__(self, hidden_dim, head_dim):
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

                retained = (
                    alpha_t.unsqueeze(1)
                    * S
                )

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

                new_info = (
                    beta_t
                    * torch.outer(
                        kt,
                        vt
                    )
                )

                S_new = (
                    corrected
                    + new_info
                )

                output = S_new.T @ qt

                batch_states.append(S_new)
                batch_outputs.append(output)

            state = torch.stack(
                batch_states
            )

            outputs.append(
                torch.stack(batch_outputs)
            )

        outputs = torch.stack(outputs)

        outputs = outputs.transpose(
            0,
            1
        )

        return self.out_proj(outputs)


# ============================================================
# Expert
# ============================================================

class Expert(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim
    ):
        super().__init__()

        self.fc1 = nn.Linear(
            hidden_dim,
            intermediate_dim
        )

        self.fc2 = nn.Linear(
            intermediate_dim,
            hidden_dim
        )

    def forward(self, x):

        return self.fc2(
            F.silu(
                self.fc1(x)
            )
        )


# ============================================================
# Tiny MoE
# ============================================================

class TinyMoE(nn.Module):

    def __init__(
        self,
        hidden_dim,
        intermediate_dim,
        num_experts=4,
        top_k=2
    ):
        super().__init__()

        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(
            hidden_dim,
            num_experts,
            bias=False
        )

        self.experts = nn.ModuleList(
            [
                Expert(
                    hidden_dim,
                    intermediate_dim
                )
                for _ in range(num_experts)
            ]
        )

    def forward(self, x):

        probabilities = F.softmax(
            self.router(x),
            dim=-1
        )

        top_values, top_indices = torch.topk(
            probabilities,
            self.top_k,
            dim=-1
        )

        top_weights = (
            top_values
            / (
                top_values.sum(
                    dim=-1,
                    keepdim=True
                )
                + 1e-8
            )
        )

        output = torch.zeros_like(x)

        for expert_id, expert in enumerate(
            self.experts
        ):

            expert_output = expert(x)

            for k in range(self.top_k):

                mask = (
                    top_indices[:, :, k]
                    == expert_id
                )

                weight = top_weights[:, :, k]

                output += (
                    expert_output
                    * mask.unsqueeze(-1)
                    * weight.unsqueeze(-1)
                )

        return output


# ============================================================
# Attention Residual
# ============================================================

class AttentionResidual(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.query = nn.Parameter(
            torch.randn(hidden_dim)
        )

    def forward(self, representations):

        keys = torch.stack(
            [
                h.mean(dim=1)
                for h in representations
            ]
        )

        scores = torch.einsum(
            "d,lbd->lb",
            self.query,
            keys
        )

        weights = F.softmax(
            scores,
            dim=0
        )

        output = torch.zeros_like(
            representations[0]
        )

        for i, h in enumerate(
            representations
        ):

            output += (
                weights[i][:, None, None]
                * h
            )

        return output


# ============================================================
# Mini K3 Layer
# ============================================================

class MiniK3Layer(nn.Module):

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

        self.moe = TinyMoE(
            hidden_dim,
            intermediate_dim
        )

    def forward(self, x):

        x = x + self.kda(
            self.norm1(x)
        )

        x = x + self.moe(
            self.norm2(x)
        )

        return x


# ============================================================
# Mini K3
# ============================================================

class MiniK3(nn.Module):

    def __init__(
        self,
        vocab_size,
        hidden_dim=32,
        head_dim=16,
        intermediate_dim=64,
        num_layers=3
    ):
        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            hidden_dim
        )

        self.layers = nn.ModuleList(
            [
                MiniK3Layer(
                    hidden_dim,
                    head_dim,
                    intermediate_dim
                )
                for _ in range(num_layers)
            ]
        )

        self.attn_res = AttentionResidual(
            hidden_dim
        )

        self.norm = nn.LayerNorm(
            hidden_dim
        )

        self.lm_head = nn.Linear(
            hidden_dim,
            vocab_size,
            bias=False
        )

    def forward(self, tokens):

        x = self.embedding(tokens)

        representations = [x]

        for layer in self.layers:

            x = layer(x)

            representations.append(x)

        x = self.attn_res(
            representations
        )

        x = self.norm(x)

        logits = self.lm_head(x)

        return logits


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
# TINY DATASET
# ============================================================

text = """
kimi k3 is a language model.
kda is a linear attention mechanism.
situ glu is a gated activation.
attention residuals combine layer representations.
mixture of experts routes tokens to experts.
kimi k3 combines efficient architectures.
""" * 20


# ============================================================
# Character vocabulary
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
print("Dataset length :", len(encoded))


# ============================================================
# Training settings
# ============================================================

sequence_length = 32

batch_size = 8

steps = 500

learning_rate = 3e-4


# ============================================================
# Batch generator
# ============================================================

def get_batch():

    starts = torch.randint(
        0,
        len(encoded) - sequence_length - 1,
        (batch_size,)
    )

    x = torch.stack(
        [
            encoded[
                i:i + sequence_length
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
# MODEL
# ============================================================

model = MiniK3(
    vocab_size=vocab_size,
    hidden_dim=32,
    head_dim=16,
    intermediate_dim=64,
    num_layers=3
).to(device)


parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(
    "Model parameters:",
    parameters
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# TRAIN
# ============================================================

losses = []

print()
print("Starting training...")
print()


for step in range(steps):

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
            f"Loss {loss.item():.4f}"
        )


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(figsize=(10, 6))

plt.plot(
    losses
)

plt.xlabel("Training step")
plt.ylabel("Cross-entropy loss")

plt.title(
    "Mini K3 Training Loss"
)

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/mini_k3_training_loss.png",
    dpi=300
)

plt.show()


# ============================================================
# TEXT GENERATION
# ============================================================

def generate(
    model,
    prompt,
    max_new_tokens=200
):

    model.eval()

    tokens = torch.tensor(
        [
            stoi[c]
            for c in prompt
            if c in stoi
        ],
        dtype=torch.long,
        device=device
    ).unsqueeze(0)

    for _ in range(
        max_new_tokens
    ):

        context = tokens[
            :, -sequence_length:
        ]

        logits = model(context)

        logits = logits[:, -1, :]

        probabilities = F.softmax(
            logits,
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

    return "".join(
        itos[int(i)]
        for i in tokens[0]
    )


# ============================================================
# GENERATE
# ============================================================

print()
print("================================")
print("GENERATED TEXT")
print("================================")
print()

result = generate(
    model,
    "kimi ",
    max_new_tokens=200
)

print(result)
