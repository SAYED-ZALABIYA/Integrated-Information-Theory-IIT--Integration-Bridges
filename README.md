# Exploring Integration in Artificial Neural Systems  
### Statistical Proxies vs Causal IIT Measures

This repository contains a research-oriented prototype that explores **information integration**
in artificial systems by connecting:

- **Statistical integration** inside neural networks  
- **Causal integration** as defined by **Integrated Information Theory (IIT)**

The project does **not** attempt to claim consciousness in AI.  
Its goal is to study **measurable bridges** between representation learning, information structure, and causal mechanisms.

---

## Conceptual Overview

Integrated Information Theory (IIT) defines **Φ (Phi)** as a measure of how much a system is
causally irreducible.

However, computing Φ directly for neural networks is intractable.

This project explores a two-track approach:

| Track | Level | What is Measured |
|-----|------|----------------|
| Phase 1 | Statistical | Total Correlation (TC) in neural activations |
| Phase 2 | Causal | True Φ from explicit Transition Probability Matrices |

The central research question:

> **Can statistical integration trends inside neural networks approximate or predict causal Φ?**

---

## Repository Structure

```text
src/
 ├── integration_proxy_tc.py    # Phase 1: TC in neural networks
 └── iit_causal_phi_tpm.py      # Phase 2: Φ from binary causal systems

data/
 ├── results_enhanced.csv
 └── epoch_data.json

assets/
 └── figures/                  # Generated visualizations
```

Phase 1 — Statistical Integration (Proxy)

File: src/integration_proxy_tc.py

A lightweight Tiny MLP trained on simple datasets:

XOR

Two Moons

Spiral

What is tracked

Total Correlation (TC) per layer

TC Sum across layers

Accuracy vs training epochs

Effects of connectivity density and residual links

Purpose

TC acts as a statistical proxy for how integrated learned representations become during training.

⚠️ Any Φ value inside this phase (if present) is a toy demonstration only,
not causal IIT Φ.
