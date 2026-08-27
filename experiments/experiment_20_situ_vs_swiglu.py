import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path

print("=" * 60)
print("EXPERIMENT 20: SwiGLU vs SiTU-GLU")
print("=" * 60)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Device:", device)


def swiglu(gate, up):
    return F.silu(gate) * up


def situ_glu(gate, up, beta1=4.0, beta2=25.0):
    gate_branch = (
        beta1
        * torch.tanh(gate / beta1)
        * torch.sigmoid(gate)
    )

    up_branch = (
        beta2
        * torch.tanh(up / beta2)
    )

    return gate_branch * up_branch


x = torch.linspace(-20, 20, 2000, device=device)

gate = x
up = x

with torch.no_grad():
    swiglu_output = swiglu(gate, up)
    situ_output = situ_glu(gate, up)


Path("results").mkdir(exist_ok=True)


# ============================================================
# GRAPH 1
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    x.cpu().numpy(),
    swiglu_output.cpu().numpy(),
    label="SwiGLU"
)

plt.plot(
    x.cpu().numpy(),
    situ_output.cpu().numpy(),
    label="SiTU-GLU"
)

plt.xlabel("Input")
plt.ylabel("Activation Output")
plt.title("SwiGLU vs SiTU-GLU")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/experiment_20_activation_comparison.png",
    dpi=200
)

plt.show()


# ============================================================
# GRAPH 2
# ============================================================

plt.figure(figsize=(12, 7))

plt.plot(
    x.cpu().numpy(),
    torch.abs(swiglu_output).cpu().numpy(),
    label="|SwiGLU|"
)

plt.plot(
    x.cpu().numpy(),
    torch.abs(situ_output).cpu().numpy(),
    label="|SiTU-GLU|"
)

plt.xlabel("Input")
plt.ylabel("Absolute Output")
plt.title("Output Magnitude: SwiGLU vs SiTU-GLU")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "results/experiment_20_output_magnitude.png",
    dpi=200
)

plt.show()


# ============================================================
# RESULTS
# ============================================================

swiglu_max = torch.max(torch.abs(swiglu_output)).item()
situ_max = torch.max(torch.abs(situ_output)).item()

swiglu_mean = torch.mean(torch.abs(swiglu_output)).item()
situ_mean = torch.mean(torch.abs(situ_output)).item()

print()
print("=" * 60)
print("NUMERICAL ANALYSIS")
print("=" * 60)

print(f"SwiGLU maximum absolute output: {swiglu_max:.4f}")
print(f"SiTU-GLU maximum absolute output: {situ_max:.4f}")

print(f"SwiGLU mean absolute output: {swiglu_mean:.4f}")
print(f"SiTU-GLU mean absolute output: {situ_mean:.4f}")


with open("results/experiment_20_results.txt", "w") as f:

    f.write("Experiment 20: SwiGLU vs SiTU-GLU\n")
    f.write("=" * 50 + "\n")

    f.write(f"SwiGLU max abs output: {swiglu_max:.6f}\n")
    f.write(f"SiTU-GLU max abs output: {situ_max:.6f}\n")

    f.write(f"SwiGLU mean abs output: {swiglu_mean:.6f}\n")
    f.write(f"SiTU-GLU mean abs output: {situ_mean:.6f}\n")


print()
print("=" * 60)
print("EXPERIMENT 20 COMPLETE")
print("=" * 60)

print()
print("Generated files:")
print("results/experiment_20_activation_comparison.png")
print("results/experiment_20_output_magnitude.png")
print("results/experiment_20_results.txt")
