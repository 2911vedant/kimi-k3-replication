import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

# ============================================================
# EXPERIMENT 21: FINAL SwiGLU vs SiTU-GLU COMPARISON
# ============================================================

print("=" * 60)
print("EXPERIMENT 21: FINAL COMPARISON")
print("=" * 60)

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)

print(f"Device: {device}")

os.makedirs("results", exist_ok=True)


# ============================================================
# ACTIVATION FUNCTIONS
# ============================================================

class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.silu(x) * x


class SiTU_GLUE(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # SiTU-style bounded gate
        gate = torch.sigmoid(x)
        return x * gate


# ============================================================
# CREATE INPUT RANGE
# ============================================================

x = torch.linspace(-20, 20, 1000).to(device)

swiglu = SwiGLU().to(device)
situ_glu = SiTU_GLUE().to(device)

with torch.no_grad():
    y_swiglu = swiglu(x)
    y_situ = situ_glu(x)


# ============================================================
# NUMERICAL ANALYSIS
# ============================================================

print()
print("=" * 60)
print("ACTIVATION ANALYSIS")
print("=" * 60)

swiglu_max = torch.max(torch.abs(y_swiglu)).item()
situ_max = torch.max(torch.abs(y_situ)).item()

swiglu_mean = torch.mean(torch.abs(y_swiglu)).item()
situ_mean = torch.mean(torch.abs(y_situ)).item()

print(f"SwiGLU maximum absolute output : {swiglu_max:.4f}")
print(f"SiTU-GLU maximum absolute output: {situ_max:.4f}")

print(f"SwiGLU mean absolute output     : {swiglu_mean:.4f}")
print(f"SiTU-GLU mean absolute output   : {situ_mean:.4f}")


# ============================================================
# PLOT 1 — ACTIVATION COMPARISON
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    x.cpu().numpy(),
    y_swiglu.cpu().numpy(),
    label="SwiGLU"
)

plt.plot(
    x.cpu().numpy(),
    y_situ.cpu().numpy(),
    label="SiTU-GLU"
)

plt.xlabel("Input")
plt.ylabel("Activation Output")
plt.title("Final SwiGLU vs SiTU-GLU Activation Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/experiment_21_final_activation.png",
    dpi=200
)

plt.show()


# ============================================================
# PLOT 2 — ABSOLUTE OUTPUT
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    x.cpu().numpy(),
    torch.abs(y_swiglu).cpu().numpy(),
    label="SwiGLU"
)

plt.plot(
    x.cpu().numpy(),
    torch.abs(y_situ).cpu().numpy(),
    label="SiTU-GLU"
)

plt.xlabel("Input")
plt.ylabel("Absolute Output")
plt.title("Final Output Magnitude Comparison")
plt.legend()
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/experiment_21_final_magnitude.png",
    dpi=200
)

plt.show()


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("FINAL INTERPRETATION")
print("=" * 60)

if situ_max < swiglu_max:
    print("SiTU-GLU produces a lower maximum activation magnitude.")

if situ_mean < swiglu_mean:
    print("SiTU-GLU produces a lower mean activation magnitude.")

reduction = (1 - situ_max / swiglu_max) * 100

print(f"Maximum-output reduction: {reduction:.2f}%")

print()
print("Generated files:")
print("results/experiment_21_final_activation.png")
print("results/experiment_21_final_magnitude.png")

print()
print("=" * 60)
print("EXPERIMENT 21 COMPLETE")
print("=" * 60)
