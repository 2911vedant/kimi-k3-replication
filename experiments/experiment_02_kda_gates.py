import torch
import matplotlib.pyplot as plt


def kda_step(S_prev, q, k, v, alpha, beta):

    d = k.shape[0]

    I = torch.eye(d, device=S_prev.device)

    retention = torch.diag(alpha)

    old_memory = retention @ S_prev

    correction = (
        I - beta * torch.outer(k, k)
    ) @ old_memory

    new_information = (
        beta * torch.outer(k, v)
    )

    S_new = correction + new_information

    output = S_new.T @ q

    return S_new, output


# --------------------------------
# Settings
# --------------------------------

torch.manual_seed(42)

device = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print("Device:", device)

sequence_length = 20
dimension = 8


# --------------------------------
# Run one experiment
# --------------------------------

def run_experiment(alpha_value, beta_value):

    S = torch.zeros(
        dimension,
        dimension,
        device=device
    )

    state_norms = []

    for t in range(sequence_length):

        q = torch.randn(
            dimension,
            device=device
        )

        k = torch.randn(
            dimension,
            device=device
        )

        v = torch.randn(
            dimension,
            device=device
        )

        # Normalize Q and K
        q = q / (torch.norm(q) + 1e-8)
        k = k / (torch.norm(k) + 1e-8)

        alpha = torch.full(
            (dimension,),
            alpha_value,
            device=device
        )

        beta = torch.tensor(
            beta_value,
            device=device
        )

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

    return state_norms


# --------------------------------
# Three experiments
# --------------------------------

high_retention = run_experiment(
    alpha_value=0.95,
    beta_value=0.50
)

low_retention = run_experiment(
    alpha_value=0.20,
    beta_value=0.50
)

strong_writing = run_experiment(
    alpha_value=0.95,
    beta_value=0.95
)


# --------------------------------
# Plot
# --------------------------------

plt.figure(figsize=(10, 6))

plt.plot(
    range(1, sequence_length + 1),
    high_retention,
    label="High retention α=0.95"
)

plt.plot(
    range(1, sequence_length + 1),
    low_retention,
    label="Low retention α=0.20"
)

plt.plot(
    range(1, sequence_length + 1),
    strong_writing,
    label="Strong writing β=0.95"
)

plt.xlabel("Token")
plt.ylabel("||S||")

plt.title(
    "Effect of α and β on KDA Memory"
)

plt.legend()

plt.grid(True)

plt.savefig(
    "results/kda_gate_comparison.png",
    dpi=300
)

plt.show()
