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

Phase 2 — Causal Integration (IIT Ground Truth)

File: src/iit_causal_phi_tpm.py

This phase isolates causality from learning.

What is implemented

Explicit binary systems (3–5 nodes)

Deterministic logical architectures (AND, OR, XOR, NAND chains)

Construction of full Transition Probability Matrices (TPMs)

Exact computation of Φ using PyPhi

Purpose

Provides a mechanistic ground truth for causal integration, fully aligned with IIT.

Example Results (Preliminary)

In initial runs, TC tends to decrease and compress as accuracy increases.

This suggests a possible relationship between learning dynamics and integration structure.

Results are exploratory and single-run at this stage.
Statistical significance is not claimed yet.

Installation
pip install -r requirements.txt

Running the Experiments
Phase 1 — Neural Integration
python src/integration_proxy_tc.py --dataset xor --epochs 40 --visualize

Phase 2 — Causal Φ
python src/iit_causal_phi_tpm.py --preset xor_nand_5
python src/iit_causal_phi_tpm.py --preset and_or_4 --state 1010

Research Direction (Next Steps)

Binarization of neural activations

Reduced-state causal abstraction of trained networks

Testing whether TC trends predict Φ under controlled reductions

Multi-run statistical validation

Key Takeaway

This project demonstrates a methodological bridge between:

Representation learning

Information theory

Causal structure

It is a step toward quantifying integration in artificial systems, not defining consciousness.
