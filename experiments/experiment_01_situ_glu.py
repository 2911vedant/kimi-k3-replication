import torch
import matplotlib.pyplot as plt


# -----------------------------
# 1. GLU
# -----------------------------
def glu(x):
    gate = torch.sigmoid(x)
    value = x
    return gate * value


# -----------------------------
# 2. SwiGLU
# -----------------------------
def swiglu(x):
    gate = torch.nn.functional.silu(x)
    value = x
    return gate * value


# -----------------------------
# 3. SiTU-GLU
# -----------------------------
def situ_glu(x, beta1=4.0, beta2=25.0):
    gate = beta1 * torch.tanh(x / beta1) * torch.sigmoid(x)
    value = beta2 * torch.tanh(x / beta2)

    return gate * value


# -----------------------------
# 4. Create input values
# -----------------------------
x = torch.linspace(-10, 100, 1000)


# -----------------------------
# 5. Calculate outputs
# -----------------------------
y_glu = glu(x)
y_swiglu = swiglu(x)
y_situ = situ_glu(x)


# -----------------------------
# 6. Print maximum SiTU-GLU
# -----------------------------
print("Maximum SiTU-GLU value:", y_situ.max().item())


# -----------------------------
# 7. Plot
# -----------------------------
plt.figure(figsize=(10, 6))

plt.plot(x.numpy(), y_glu.numpy(), label="GLU")
plt.plot(x.numpy(), y_swiglu.numpy(), label="SwiGLU")
plt.plot(x.numpy(), y_situ.numpy(), label="SiTU-GLU")

plt.xlabel("Input x")
plt.ylabel("Output")
plt.title("GLU vs SwiGLU vs SiTU-GLU")

plt.legend()
plt.grid(True)

plt.savefig("results/situ_glu_comparison.png", dpi=300)

plt.show()
