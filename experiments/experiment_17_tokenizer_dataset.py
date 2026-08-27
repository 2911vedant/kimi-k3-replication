import tiktoken
import torch


# ============================================================
# EXPERIMENT 17
# PROPER TOKENIZER + DATASET
# ============================================================

print("==========================================")
print("EXPERIMENT 17")
print("TOKENIZER + DATASET")
print("==========================================")


# ============================================================
# 1. LOAD TOKENIZER
# ============================================================

tokenizer = tiktoken.get_encoding("cl100k_base")

print()
print("Tokenizer loaded successfully.")
print("Tokenizer:", "cl100k_base")


# ============================================================
# 2. TRAINING TEXT
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

This experiment demonstrates tokenization and dataset
preparation before training the Mini K3 model.
"""


# ============================================================
# 3. TOKENIZE
# ============================================================

tokens = tokenizer.encode(text)


print()
print("Original characters:", len(text))
print("Number of tokens    :", len(tokens))


# ============================================================
# 4. SHOW FIRST TOKENS
# ============================================================

print()
print("First 30 token IDs:")
print(tokens[:30])


# ============================================================
# 5. DECODE TOKENS
# ============================================================

decoded = tokenizer.decode(
    tokens[:30]
)


print()
print("Decoded first 30 tokens:")
print(decoded)


# ============================================================
# 6. CONVERT TO PYTORCH
# ============================================================

data = torch.tensor(
    tokens,
    dtype=torch.long
)


print()
print("PyTorch tensor:")
print(data.shape)

print()
print("Tensor dtype:")
print(data.dtype)


# ============================================================
# 7. CREATE SEQUENCES
# ============================================================

sequence_length = 32


x = data[
    :sequence_length
]

y = data[
    1:sequence_length + 1
]


print()
print("==========================================")
print("NEXT TOKEN PREDICTION")
print("==========================================")


print()
print("Input sequence:")
print(
    tokenizer.decode(
        x.tolist()
    )
)


print()
print("Target sequence:")
print(
    tokenizer.decode(
        y.tolist()
    )
)


# ============================================================
# 8. VERIFY NEXT TOKEN RELATIONSHIP
# ============================================================

print()
print("Token prediction pairs:")
print("------------------------------------------")


for i in range(
    min(10, len(x))
):

    current_token = x[i].item()

    next_token = y[i].item()

    current_text = tokenizer.decode(
        [current_token]
    )

    next_text = tokenizer.decode(
        [next_token]
    )

    print(
        f"{i:2d}: "
        f"{repr(current_text):15s}"
        " -> "
        f"{repr(next_text)}"
    )


# ============================================================
# 9. RANDOM BATCH GENERATOR
# ============================================================

def get_batch(
    data,
    batch_size=4,
    sequence_length=32
):

    starts = torch.randint(
        0,
        len(data) - sequence_length - 1,
        (batch_size,)
    )


    x_batch = torch.stack(
        [
            data[
                start:
                start + sequence_length
            ]

            for start in starts
        ]
    )


    y_batch = torch.stack(
        [
            data[
                start + 1:
                start + sequence_length + 1
            ]

            for start in starts
        ]
    )


    return x_batch, y_batch


# ============================================================
# 10. TEST BATCH
# ============================================================

x_batch, y_batch = get_batch(
    data,
    batch_size=4,
    sequence_length=32
)


print()
print("==========================================")
print("BATCH TEST")
print("==========================================")


print()
print("Input batch shape:")
print(x_batch.shape)


print()
print("Target batch shape:")
print(y_batch.shape)


# ============================================================
# 11. DISPLAY BATCH
# ============================================================

for i in range(
    x_batch.shape[0]
):

    print()
    print(
        f"Batch {i + 1}"
    )

    print(
        "Input :",
        tokenizer.decode(
            x_batch[i].tolist()
        )
    )

    print(
        "Target:",
        tokenizer.decode(
            y_batch[i].tolist()
        )
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("==========================================")
print("EXPERIMENT 17 COMPLETE")
print("==========================================")

print()
print("We now have:")

print("✓ Subword tokenizer")

print("✓ Token IDs")

print("✓ PyTorch dataset")

print("✓ Input sequences")

print("✓ Next-token targets")

print("✓ Random training batches")

print()
print("Ready for the next training experiment.")
