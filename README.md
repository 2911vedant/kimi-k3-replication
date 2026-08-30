# Kimi K3 Replication

A research-oriented replication and analysis of key architectural components of the Kimi K3 language model, with a focus on SiTU-GLU, KDA, Mixture-of-Experts (MoE), and related components.

## Project Overview

This project investigates and reproduces important components discussed in the Kimi K3 architecture. The experiments progressively analyze the activation functions, attention mechanisms, model blocks, routing behavior, and training characteristics.

## Main Components

- **SiTU-GLU** — Analysis and implementation of the SiTU-GLU activation function.
- **KDA** — Investigation of Kimi's KDA attention mechanism.
- **SwiGLU vs SiTU-GLU** — Numerical and visual comparison of activation behavior.
- **Mini Kimi K3** — Lightweight implementation of the Kimi K3 architecture.
- **Mixture-of-Experts (MoE)** — Expert routing and utilization analysis.
- **Tokenizer & Dataset** — Tokenization and dataset preparation experiments.
- **Training** — Small-scale model training and loss analysis.

## Experiments

The `experiments/` directory contains the individual experiments:

```text
experiment_01_situ_glu.py
experiment_02_kda_gates.py
experiment_03_kda_projections.py
experiment_04_kda_head.py
experiment_05_kda_situ.py
experiment_06_multi_layer.py
experiment_07_attnres.py
experiment_08_moe.py
experiment_09_mini_k3.py
experiment_10_train.py
experiment_11_swiglu_vs_situ.py
experiment_12_figure4.py
experiment_13_verify_situ.py
experiment_14_kda.py
experiment_15_mini_k3_block.py
experiment_16_mini_k3_lm.py
experiment_17_tokenizer_dataset.py
experiment_18_train_token_model.py
experiment_19_baseline_comparison.py
experiment_20_situ_vs_swiglu.py
experiment_21_final_comparison.py
