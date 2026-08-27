import torch
import matplotlib.pyplot as plt


# ==========================================
# KDA - Core Delta Rule
# ==========================================

def kda_step(S_prev, q, k, v, alpha, beta):
    """
    One simplified KDA state update.

    S_prev : previous memory state
    q      : query
    k      : key
    v      : value
    alpha  : channel-wise retention
    beta   : write strength
    """

    d = k.shape[0]

    # Identity matrix
    I = torch.eye(d, device=S_prev.device)

    # Channel-wise retention
    retention = torch.diag(alpha)

    # Retain part of previous memory
    old_memory = retention @ S_prev

    # Delta-rule correction
    correction = (I - beta * torch.outer(k, k)) @ old_memory

    # Write new key-value information
    new_information = beta * torch.outer(k, v)

    # New recurrent state
    S_new = correction + new_information

    # Read from memory using the query
    output = S_new.T @ q

    return S_new, output


# ==========================================
# Experiment settings
# ==========================================

torch.manual_seed(42)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


# Small dimensions for our experiment
sequence_length = 20
dimension = 8


# ==========================================
# Initialize recurrent state
# ==========================================

S = torch.zeros(
    dimension,
    dimension,
    device=device
)


state_norms = []
output_norms = []


# ==========================================
# Process tokens one by one
# ==========================================

for t in range(sequence_length):

    # Random Q, K, V for demonstration
    q = torch.randn(dimension, device=device)
    k = torch.randn(dimension, device=device)
    v = torch.randn(dimension, device=device)

    # Normalize Q and K
    q = q / (torch.norm(q) + 1e-8)
    k = k / (torch.norm(k) + 1e-8)

    # Channel-wise retention
    alpha = torch.sigmoid(
        torch.randn(dimension, device=device)
    )

    # Write strength
    beta = torch.sigmoid(
        torch.randn((), device=device)
    )

    # KDA update
    S, output = kda_step(
        S,
        q,
        k,
        v,
        alpha,
        beta
    )

    state_norms.append(
        torch.norm(S).item()
    )

    output_norms.append(
        torch.norm(output).item()
    )

    print(
        f"Token {t + 1:02d} | "
        f"beta={beta.item():.3f} | "
        f"state_norm={state_norms[-1]:.3f}"
    )


# ==========================================
# Plot state evolution
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, sequence_length + 1),
    state_norms,
    marker="o"
)

plt.xlabel("Token")
plt.ylabel("||S||")
plt.title("KDA Recurrent State Evolution")

plt.grid(True)

plt.savefig(
    "results/kda_state_evolution.png",
    dpi=300
)

plt.show()


# ==========================================
# Plot output evolution
# ==========================================

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, sequence_length + 1),
    output_norms,
    marker="o"
)

plt.xlabel("Token")
plt.ylabel("||Output||")
plt.title("KDA Output Evolution")

plt.grid(True)

plt.savefig(
    "results/kda_output_evolution.png",
    dpi=300
)

plt.show()
