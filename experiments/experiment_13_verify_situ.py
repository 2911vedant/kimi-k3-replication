import torch
import matplotlib.pyplot as plt


# ============================================================
# EXPERIMENT 13
# NUMERICAL VERIFICATION OF SiTU-GLU
# ============================================================

torch.set_default_dtype(torch.float64)


# ============================================================
# PARAMETERS FROM KIMI K3
# ============================================================

beta1 = 4.0
beta2 = 25.0

theoretical_bound = beta1 * beta2


print()
print("==========================================")
print("SiTU-GLU THEORETICAL VERIFICATION")
print("==========================================")

print("beta1 =", beta1)
print("beta2 =", beta2)

print(
    "Theoretical maximum =",
    theoretical_bound
)


# ============================================================
# FUNCTIONS
# ============================================================

def sigmoid(x):
    return torch.sigmoid(x)


def swiglu(x):

    return (
        x
        * sigmoid(x)
    )


def situ_glu(x):

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
# LARGE POSITIVE INPUTS
# ============================================================

values = [
    1,
    4,
    10,
    25,
    50,
    100,
    250,
    500,
    1000
]


print()
print("Large positive inputs:")
print("------------------------------------------")

for value in values:

    x = torch.tensor(float(value))

    s = swiglu(x).item()

    t = situ_glu(x).item()

    print(
        f"x={value:>5} | "
        f"SwiGLU={s:>12.4f} | "
        f"SiTU-GLU={t:>12.4f}"
    )


# ============================================================
# CHECK THE BOUND
# ============================================================

x = torch.linspace(
    -100,
    100,
    100000
)

situ_values = situ_glu(x)

maximum = torch.max(
    torch.abs(situ_values)
).item()

print()
print("------------------------------------------")

print(
    "Observed maximum |SiTU-GLU| =",
    maximum
)

print(
    "Theoretical bound           =",
    theoretical_bound
)

print(
    "Bound satisfied:",
    maximum <= theoretical_bound
)


# ============================================================
# PLOT LARGE RANGE
# ============================================================

plt.figure(figsize=(11, 6))

plt.plot(
    x.numpy(),
    situ_values.numpy(),
    label="SiTU-GLU"
)

plt.axhline(
    theoretical_bound,
    linestyle="--",
    label="Theoretical upper bound"
)

plt.axhline(
    -theoretical_bound,
    linestyle="--",
    label="Theoretical lower bound"
)

plt.xlabel(
    "Input x"
)

plt.ylabel(
    "SiTU-GLU(x)"
)

plt.title(
    "SiTU-GLU Boundedness Verification"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/situ_boundedness.png",
    dpi=300
)

plt.show()


# ============================================================
# COMPARE GROWTH
# ============================================================

x_growth = torch.linspace(
    0,
    100,
    2000
)

swiglu_values = swiglu(
    x_growth
)

situ_values = situ_glu(
    x_growth
)


plt.figure(figsize=(11, 6))

plt.plot(
    x_growth.numpy(),
    swiglu_values.numpy(),
    label="SwiGLU"
)

plt.plot(
    x_growth.numpy(),
    situ_values.numpy(),
    label="SiTU-GLU"
)

plt.axhline(
    theoretical_bound,
    linestyle="--",
    label="SiTU-GLU bound"
)

plt.xlabel(
    "Input x"
)

plt.ylabel(
    "Activation output"
)

plt.title(
    "SwiGLU vs SiTU-GLU Growth"
)

plt.legend()

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/swiglu_vs_situ_growth.png",
    dpi=300
)

plt.show()


# ============================================================
# FINAL
# ============================================================

print()
print("==========================================")
print("EXPERIMENT 13 COMPLETE")
print("==========================================")

print()
print("Results saved to:")

print(
    "results/situ_boundedness.png"
)

print(
    "results/swiglu_vs_situ_growth.png"
)
