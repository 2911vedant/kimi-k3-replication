import torch
import matplotlib.pyplot as plt


# ============================================================
# KIMI K3 FIGURE 4 REPLICATION
# ============================================================

torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------
# Input range used for the comparison
# ------------------------------------------------------------

x = torch.linspace(-10, 100, 2000)


# ------------------------------------------------------------
# Sigmoid
# ------------------------------------------------------------

def sigmoid(x):
    return torch.sigmoid(x)


# ------------------------------------------------------------
# GLU
# f(x) = sigmoid(x) * x
# ------------------------------------------------------------

def glu(x):
    return sigmoid(x) * x


# ------------------------------------------------------------
# SwiGLU
# f(x) = x * sigmoid(x)
# ------------------------------------------------------------

def swiglu(x):
    return x * sigmoid(x)


# ------------------------------------------------------------
# SiTU-GLU
#
# beta1 = 4
# beta2 = 25
# ------------------------------------------------------------

def situ_glu(x):

    beta1 = 4.0
    beta2 = 25.0

    gate = (
        beta1
        * torch.tanh(x / beta1)
        * sigmoid(x)
    )

    up = (
        beta2
        * torch.tanh(x / beta2)
    )

    return gate * up


# ============================================================
# CALCULATE
# ============================================================

y_glu = glu(x)

y_swiglu = swiglu(x)

y_situ = situ_glu(x)


# ============================================================
# PRINT IMPORTANT VALUES
# ============================================================

print()
print("==========================================")
print("KIMI K3 FIGURE 4")
print("==========================================")

for value in [-10, -5, 0, 1, 4, 10, 25, 50, 75, 100]:

    xx = torch.tensor(float(value))

    print(
        f"x = {value:>4} | "
        f"GLU = {glu(xx).item():>10.4f} | "
        f"SwiGLU = {swiglu(xx).item():>10.4f} | "
        f"SiTU-GLU = {situ_glu(xx).item():>10.4f}"
    )


# ============================================================
# MAIN FIGURE
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    x.numpy(),
    y_glu.numpy(),
    label="GLU",
    linewidth=2
)

plt.plot(
    x.numpy(),
    y_swiglu.numpy(),
    label="SwiGLU",
    linewidth=2
)

plt.plot(
    x.numpy(),
    y_situ.numpy(),
    label="SiTU-GLU",
    linewidth=2
)

plt.xlabel("Input x")

plt.ylabel("Scalar response")

plt.title(
    "Kimi K3 Figure 4 — GLU vs SwiGLU vs SiTU-GLU"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/kimi_k3_figure4_replication.png",
    dpi=300
)

plt.show()


# ============================================================
# NEAR ORIGIN — INSET STYLE PLOT
# ============================================================

mask = x <= 10

plt.figure(figsize=(10, 6))

plt.plot(
    x[mask].numpy(),
    y_glu[mask].numpy(),
    label="GLU",
    linewidth=2
)

plt.plot(
    x[mask].numpy(),
    y_swiglu[mask].numpy(),
    label="SwiGLU",
    linewidth=2
)

plt.plot(
    x[mask].numpy(),
    y_situ[mask].numpy(),
    label="SiTU-GLU",
    linewidth=2
)

plt.xlabel("Input x")

plt.ylabel("Scalar response")

plt.title(
    "Near-Origin Comparison"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/kimi_k3_figure4_near_origin.png",
    dpi=300
)

plt.show()


print()
print("==========================================")
print("FIGURE 4 REPLICATION COMPLETE")
print("==========================================")

print()
print("Saved:")
print("results/kimi_k3_figure4_replication.png")
print("results/kimi_k3_figure4_near_origin.png")
