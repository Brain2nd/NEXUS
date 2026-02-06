# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an academic paper repository for **LASER: A High-Fidelity Spike Representation SNN Framework With Surrogate-Free Training**, an ICML 2026 submission. This is a LaTeX-based academic paper project, not a software implementation.

## Build Commands

Compile the paper using standard LaTeX workflow:

```bash
pdflatex example_paper.tex
bibtex example_paper
pdflatex example_paper.tex
pdflatex example_paper.tex
```

Clean auxiliary files:

```bash
rm -f *.aux *.log *.bbl *.blg *.out
```

## Repository Structure

- `example_paper.tex` - Main LaTeX document that includes all sections
- `section/` - Primary paper content (used in compilation)
  - `01introduction.tex` - Introduction
  - `02relatedwork.tex` - Related Work
  - `03methods.tex` - Methodology (BSE, ASNC, STE)
  - `04experiment.tex` - Experiments and Results
  - `05conclusion.tex` - Conclusion
  - `06appendix.tex` - Mathematical proofs
- `section001/`, `section002/` - Alternative/draft versions (not included in compilation)
- `figures/` - Paper figures (PDF/JPEG format)
- `icml2026.sty`, `icml2026.bst` - ICML 2026 conference style files
- `example_paper.bib` - Primary bibliography file (referenced in main document)

## Paper Architecture

LASER proposes a three-component framework for high-fidelity ANN-to-SNN conversion:

1. **Bit Spike Encoding (BSE)**: Deterministic bidirectional mapping between N-step spike sequences and N-bit numerical values. Achieves 10^11-10^18 times lower error than rate coding/TTFS.

2. **Adaptive Spiking Neural Codec (ASNC)**: Piecewise approximator for nonlinear functions (ReLU, Softmax, SiLU, etc.) that confines approximation error to ~8.7×10^-7 MSE.

3. **STE-based Training**: Straight-Through Estimator backpropagation enabling surrogate-free training with stable convergence.

## Key Mathematical Content

- Equations defined in `section/03methods.tex`: membrane potential evolution, soma computation, ASNC codec mapping, adaptive splitting, weighted loss
- Formal proofs in `section/06appendix.tex`: encoder-decoder bijection theorem, entropy preservation, ASNC error bounds
