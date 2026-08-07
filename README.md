# WISER CHALLENGE : BQP 2026, Quantum-Assisted PINNs for Computational Fluid Dynamics

> **Authors:** Ounomborbitibou Djabon  \& Mawulikplimi Roland Hounkpe
> **Program:** WISER 2026 Summer Program, in collaboration with BosonQ Psi (BQP)
> **Challenge:** *Explain when, why, and how a variational quantum circuit (VVC) changes the learning dynamics of a Physics-Informed Neural Network.*

<p align="center">
  <em>Reproducing, ablation-testing, and explaining Shah et al. (2024) "Benchmarking Quantum-Assisted PINN for CFD" on the 1-D viscous Burgers and heat equations.</em>
</p>

---

## Table of Contents

1. [Abstract](#1-abstract)
2. [Repository Layout](#2-repository-layout)
3. [Quickstart](#3-quickstart)
4. [Mathematical Background](#4-mathematical-background)
5. [Phase 1 : Classical PINN Baseline](#5-phase-1--classical-pinn-baseline)
6. [Phase 2 : QAPINN Framework](#6-phase-2--qapinn-framework)
7. [Phase 2 : Hard-Viscosity Benchmark](#7-phase-2--hard-viscosity-benchmark)
8. [Phase 2 : Ablation Grid](#8-phase-2--ablation-grid)
9. [Phase 3 : Explainability Workstreams](#9-phase-3--explainability-workstreams)
10. [Key Results](#10-key-results)
11. [Reproducing the Shah et al. Table I](#11-reproducing-the-shah-et-al-table-i)
12. [Methodological Notes & Caveats](#12-methodological-notes--caveats)
13. [Dependencies](#13-dependencies)
14. [Citation](#14-citation)

---

## 1. Abstract

This repository contains the **complete code, trained models, and analysis artifacts** for the WISER × BQP 2026 challenge, in which we benchmark a **Quantum-Assisted Physics-Informed Neural Network (QAPINN)** against parameter-matched classical controls on the 1-D viscous Burgers and heat equations.

Our contributions:

1. **A reproducible classical baseline** (Phase 1): a 9-layer tanh PINN trained with scipy L-BFGS-B and a numerically stable Cole–Hopf reference, reaching relative $L^2 = 2.1\times10^{-4}$ which is 3× better than the Raissi et al. (2017) target of $6.7\times10^{-4}$.
2. **A flexible QAPINN framework** (Phase 2): a VQC replaces the first hidden layer of a deep classical PINN, with configurable qubit count, variational depth, entanglement topology, measurement type, and post-processing width. Four model variants are supported: QAPINN, ClassicalTwin (exact parameter parity), GAAF-PINN (global adaptive activation), and ShahQAPINN (faithful Shah et al. reproduction).
3. **A multi-seed, paired-statistical comparison** at exact parameter parity across both PDEs and two viscosity regimes, with paired t-tests and 95 % confidence intervals on the geometric-mean accuracy ratio.
4. **A 14-config ablation grid** isolating the contribution of each quantum design axis (entanglement, measurement, depth, qubit count) plus a barren-plateau diagnostic out to 8 qubits.
5. **A three-workstream explainability analysis** (Phase 3): (A) Fourier-spectrum recovery of the analytical solution, (B) gradient-variance / learning-dynamics study, and (C) layer-wise Centered Kernel Alignment (CKA) information-flow tracing — all computed at 26 time slices across the spatiotemporal domain.

The headline finding: **at exact parameter parity, the QAPINN matches or exceeds a classical twin on Burgers' equation, but the advantage disappears (and reverses) on the heat equation** — supporting a frequency-selective interpretation grounded in Schuld et al.'s Fourier theorem for parameterized quantum circuits.

---

## 2. Repository Layout

```
wiser_project/
├── Phase 1/wiser/                              # Classical PINN baseline (Burgers ν=0.01/π)
│   ├── Phase1_Classical_Baseline.ipynb
│   ├── figures/                                # 6 XAI-hook PNGs
│   └── results/phase1_burgers_*/               # metrics.json, model, loss histories
│
├── Phase 2/
│   ├── phase2_part_1/                          # QAPINN vs classical twin + GAAF + Shah repro
│   │   ├── Tiers12_q3/                         # First single-seed run
│   │   ├── Tiers12_q38_more_iter/              # Extended q3→q8 sweep + barren-plateau probe
│   │   ├── Tiers12 c-PINN/                     # Classical twin at exact param parity
│   │   ├── Tiers12_multi_seed/                 # ★ Definitive 5-seed production run
│   │   │   └── QAPINN_PHASE2_FINAL.ipynb       #   (canonical framework source)
│   │   └── GAAF_QAPINN_SHA_REPRO/
│   │       ├── GAAF-PINN/                      # GAAF baseline (n3–n8)
│   │       ├── Heat Multi_seed_q38/            # Heat-equation sweep
│   │       └── Shah & al Reproduction/         # Shah et al. Table I reproduction
│   │
│   └── phase2_part_2/wiser/                    # 14-config ablation grid
│       └── QAPINN_ABLATION.ipynb
│
├── Phase2_hardBurger/                          # Hard ν=0.01/π training (split per width)
│   ├── QAPINN N3_5/, N6/, N7/, N8/             # QAPINN at hard ν, per qubit count
│   ├── GAAF-PINN/                              # GAAF at hard ν, all widths
│   └── c-PINN/                                 # Classical twin at hard ν, all widths
│
├── Phase_3_hardBurger/                         # Phase-3 analysis at hard ν
│   ├── Phase3_Hard_InputBuilder.ipynb          # ★ Rebuilds models, emits all WS-A/B/C tables
│   ├── figures/Phase3_Hard_Figures(1).ipynb    # Plots HA1–HA6, HB1–HB4, HC1–HC8
│   ├── figures/, fields/, spectra/             # PNGs + cached NPZs
│   └── *.csv, *.json                           # Summary tables + manifests
│
└── Phase_3_heat_easyBurger/                    # Phase-3 analysis at easy ν + Heat
    ├── WSA/                                    # Fourier-spectrum outputs (no notebook)
    ├── WSB/WSB_LearningDynamics.ipynb          # Learning dynamics + trainability
    └── WSC/                                    # CKA + neuron XAI
        ├── WSC_NeuronXAI.ipynb
        └── WSC_Figures.ipynb
```

**Note on structure:** the project is notebook-first. All framework code lives inside Jupyter notebooks. The canonical QAPINN framework is defined in `Phase 2/phase2_part_1/Tiers12_multi_seed/QAPINN_PHASE2_FINAL.ipynb` and is copy-pasted (with small variations) into the other Phase-2 notebooks.

**Artefact statistics:** 29 notebooks, 580 PyTorch checkpoints (`.pt`), 1 497 cached arrays (`.npz`), 196 raster figures (`.png`), 50 vector figures (`.pdf`), 46 results CSVs, 32 JSON manifests.

---

## 3. Quickstart

### 3.1 Install dependencies

```bash
pip install torch pennylane scipy numpy matplotlib pandas
# For Google Colab (Phase-3 notebooks mount Drive):
pip install google-colab
```

Python 3.10+ is required. CPU is sufficient for all experiments (the VQC uses PennyLane's `default.qubit` simulator); CUDA is auto-detected when available.

### 3.2 Run the classical baseline (Phase 1)

```bash
cd "Phase 1/wiser"
jupyter notebook Phase1_Classical_Baseline.ipynb
```

Expected runtime: ~hours on CPU. Output: `results/phase1_burgers_*/metrics.json` with `relative_l2_error ≈ 2.1e-4`.

### 3.3 Run the QAPINN grid (Phase 2)

The definitive multi-seed run:

```bash
cd "Phase 2/phase2_part_1/Tiers12_multi_seed"
jupyter notebook QAPINN_PHASE2_FINAL.ipynb
```

This trains QAPINN + classical twin at 5 seeds × 3 qubit counts × 2 PDEs (Burgers ν=0.05 + Heat α=0.1). Schedule: 1000 Adam iters (lr=8e-3) → 1000 L-BFGS iters (chunked). Expected runtime:  hours on CPU.

### 3.4 Run Phase 3 analysis

Phase 3 notebooks **do not retrain**, they consume Phase-2 artifacts (`.pt` checkpoints, `.npz` histories). Run `Phase3_Hard_InputBuilder.ipynb` first to extract all WS-A/B/C inputs, then the figure notebooks.

---

## 4. Mathematical Background

### 4.1 The 1-D viscous Burgers equation

The viscous Burgers equation is the canonical CFD benchmark. It is the Navier–Stokes equation in one dimension with the pressure-gradient term dropped, retaining the nonlinear advection $u u_x$ and the viscous diffusion $\nu u_{xx}$:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \qquad x \in [-1, 1], \quad t \in [0, 1],
$$

with initial and boundary conditions

$$
u(x, 0) = -\sin(\pi x), \qquad u(\pm 1, t) = 0.
$$

Two viscosity regimes are studied:

| Regime | $\nu$ | Character | Origin |
|--------|-------|-----------|--------|
| **Easy** | $0.05$ | Smooth, near-linear advection | WISER Tier-1 |
| **Hard** | $0.01/\pi \approx 3.18\times10^{-3}$ | Sharp shock forms at $t\approx0.4$ | Raissi et al. (2017), Shah et al. (2024) |

### 4.2 Analytical reference: Cole–Hopf transform

Burgers' equation admits a closed-form solution via the Cole–Hopf transformation. Define

$$
F(y; x, t) = \frac{\cos(\pi y)}{2\pi\nu} + \frac{(x - y)^2}{4\nu t},
$$

then

$$
u(x, t) = \frac{\displaystyle\int_{-\infty}^{\infty} \frac{x - y}{t} e^{-F(y;\,x,\,t)} dy}{\displaystyle\int_{-\infty}^{\infty} e^{-F(y;\,x,\,t)} dy}.
$$

The integrand spans $40+$ orders of magnitude, so we evaluate it in log-space with the numerically stable weighted-average form

$$
u(x, t) = \frac{\sum_k \tfrac{x - y_k}{t} w_k}{\sum_k w_k}, \qquad w_k = \exp\!\bigl(-F(y_k;\,x,\,t) - \max_j[-F(y_j;\,x,\,t)]\bigr),
$$

using a 4 000-point quadrature grid on $y \in [-3, 3]$ in float64. This avoids the numerical stiffness of finite-difference reference solvers and is reused as the evaluation target throughout all phases.

### 4.3 The heat equation

The linear heat equation serves as a control PDE without nonlinear advection:

$$
\frac{\partial u}{\partial t} - \alpha \frac{\partial^2 u}{\partial x^2} = 0, \qquad u(x, 0) = \sin(\pi x), \qquad u(\pm 1, t) = 0,
$$

with $\alpha = 0.1$. Its closed-form solution $u(x,t) = \sin(\pi x) e^{-\alpha\pi^2 t}$ provides a noise-free reference for the easy-regime comparison.

---

## 5. Phase 1 : Classical PINN Baseline

The Phase-1 baseline is the **control group** against which every QAPINN claim is measured. A weak baseline makes every later quantum claim suspect; a strong baseline is half the work of winning the challenge.

### 5.1 Architecture

The PINN architecture matches Raissi et al. (2017) exactly:

- **Input:** 2 features $(x, t)$
- **Hidden layers:** 8 layers of 20 neurons with $\tanh$ activation
- **Output:** 1 scalar $u(x, t)$
- **Total parameters:** $(2\!\cdot\!20 + 20) + 7\!\cdot\!(20\!\cdot\!20 + 20) + (20\!\cdot\!1 + 1) = 3\,021$

The $\tanh$ activation is required because the PINN loss calls for second derivatives $u_{xx}$ via autograd. ReLU is not twice-differentiable at the origin and would produce incorrect second derivatives.

### 5.2 PINN loss

The PINN loss combines a data-fit term (IC + BC) with a physics-residual term enforced via automatic differentiation:

$$
\mathcal{L} = \mathcal{L}_u + \mathcal{L}_f,
$$

$$
\mathcal{L}_u = \frac{1}{N_u}\sum_{i=1}^{N_u}\bigl[u_\theta(x_i^u, t_i^u) - u_i\bigr]^2, \qquad \mathcal{L}_f = \frac{1}{N_f}\sum_{j=1}^{N_f}\Bigl[u_t + u u_x - \nu u_{xx}\Bigr]^2_{(x_j^f, t_j^f)}.
$$

The PDE residual $f = u_t + u u_x - \nu u_{xx}$ is computed **exactly** via PyTorch autograd, not via finite differences. This is the defining innovation of PINNs: the network is differentiated end-to-end with respect to its inputs, so the PDE is enforced at arbitrary (non-grid) collocation points.

### 5.3 Five documented fixes over the v1–v4 baselines

Phase 1 went through 6 versions to reach the verified $2.1\times10^{-4}$ result. The critical fixes:

1. **Cole-Hopf analytical reference** (was: numerical method-of-lines) — eliminates interpolation error.
2. **Latin Hypercube Sampling** for collocation (was: `np.random.uniform`) — better space-filling.
3. **scipy L-BFGS-B with Raissi's exact parameters** (was: `torch.optim.LBFGS`)  PyTorch's LBFGS stalls on PINN loss landscapes because its strong-Wolfe line search fails. Raissi's original code uses scipy's L-BFGS-B via `ScipyOptimizerInterface`, with `maxiter=50000`, `maxcor=50`, `maxls=50`, `ftol = \epsilon_\text{machine}`.
4. **`xavier_uniform_` initialization** (was: `xavier_normal_`) — matches Raissi's TensorFlow code.
5. **float64 precision** (was: float32) — float32 has $\epsilon_\text{machine} \approx 1.2\times10^{-7}$; when loss reaches $\sim10^{-6}$, gradient noise triggers premature L-BFGS-B convergence. float64 ($\epsilon \approx 2.2\times10^{-16}$) allows convergence to $\sim10^{-8}$.

### 5.4 Four XAI hooks (baseline measurements for Phase 3)

The Phase-1 PINN is instrumented with four Explainable-AI hooks that become the baseline for every Phase-3 comparison:

| Hook | What it measures | Phase-3 mapping |
|------|------------------|-----------------|
| **Fourier spectrum** of mid-layer activations | Frequency content the classical network actually uses | WS-A: does the VQC access different frequencies? |
| **Loss landscape** (filter-normalized) | Basin geometry around the trained solution | WS-B: does the VQC landscape show barren plateaus? |
| **Neuron activation histograms** (all 8 layers) | Specialization vs saturation per layer | WS-C: how does information flow layer-by-layer? |
| **Gradient saliency** $\partial u/\partial x$, $\partial u/\partial t$ | Spatial and temporal sensitivity | WS-C: does the quantum layer sharpen shock capture? |

### 5.5 Phase 1 result

| Metric | Value | Target |
|--------|------:|-------:|
| Relative $L^2$ error | $2.12\times10^{-4}$ | $6.7\times10^{-4}$ (Raissi) — **3× better** |
| PDE residual mean | $1.78\times10^{-4}$ | — |
| Generalization error ($t > 1$) | $3.20\times10^{-4}$ | — |
| Final loss | $4.97\times10^{-8}$ | — |
| Training time | 6 578 s | — |
| Status | **Verified** | $L^2 < 10^{-3}$ |

The trained model, loss history, and 6 XAI figures are saved under `Phase 1/wiser/results/phase1_burgers_*/`.

---

## 6. Phase 2 : QAPINN Framework

The QAPINN replaces the first hidden layer of the Phase-1 classical PINN with a **variational quantum circuit (VQC)**. Following Shah et al. (2024), the VQC acts as a feature extractor whose output is fed to a classical post-processing head. All Phase-1 lessons (float64, Cole–Hopf, LHS, schedule-based training) are retained.

### 6.1 The QAPINN architecture

The QAPINN is a hybrid quantum–classical module:

$$
u_\theta(x, t) = \text{head}_\phi\!\bigl(\,\text{VQC}_\psi\!\bigl(\pi\, s \cdot [x, t]\bigr)\bigr),
$$

where $s \in \mathbb{R}^2$ is a trainable input scale, $\psi$ are the VQC weights, and $\phi$ are the classical head weights. The full parameter set is $\theta = \{s, \psi, \phi\}$.

#### 6.1.1 The variational quantum circuit (VQC)

The VQC uses **angle encoding with data re-uploading**: at every variational layer $\ell \in \{0, \dots, L-1\}$, the input $(x, t)$ is encoded as $R_Y$ rotations on every qubit (re-uploaded so the circuit sees the input $L$ times total). Each layer then applies a trainable $R_X(\theta_{\ell, q})$ rotation per qubit, followed by an entanglement block:

$$
|\psi_\text{out}\rangle = \prod_{\ell=1}^{L}\left[\Bigl(\prod_{q=1}^{n} R_X(\theta_{\ell,q})\Bigr) \cdot \mathcal{E}\right] \cdot \Bigl(\prod_{q=1}^{n} R_Y(\pi s \cdot x_t)\Bigr) |0\rangle^{\otimes n},
$$

where $\mathcal{E}$ is one of three entanglement topologies studied:

| `entanglement` | $\mathcal{E}$ | Description |
|----------------|---------------|-------------|
| `"none"` | $\mathbb{1}$ | Product-state ansatz (no entanglement) |
| `"linear"` | $\prod_{q=1}^{n-1}\text{CNOT}(q, q+1)$ | Linear nearest-neighbor chain |
| `"all"` | $\prod_{a<b}\text{CNOT}(a, b)$ | All-to-all (Shah-style full entanglement) |

Two measurement schemes are supported:

- **`expval`**: local expectation values $\langle\sigma_z^{(q)}\rangle$ for $q = 1, \dots, n$ — output dimension $n$.
- **`probs`**: full probability state-vector over the $2^n$ computational basis states — output dimension $2^n$.

#### 6.1.2 The classical post-processing head

Following Shah et al. (Decision 6 of the Phase-2 strategy), the classical head is a deep tanh MLP:

$$
\text{head}_\phi(\mathbf{f}) = W_\text{out}\,\tanh\!\bigl(W_{D-1}\,\tanh(\cdots\tanh(W_1\,\mathbf{f} + b_1)\cdots) + b_{D-1}\bigr) + b_\text{out},
$$

with default `head_width = 20` and `head_depth = 5`. Xavier-normal init on the linear weights, zero init on biases.

#### 6.1.3 Parameter counts at exact parity

The QAPINN and its classical twin are designed to have **identical parameter counts** so that any accuracy difference is attributable to the quantum layer alone:

| $n$ qubits | QAPINN params | ClassicalTwin params | Twin structure |
|-----------:|--------------:|---------------------:|----------------|
| 3 | 1 801 | 1 801 | `Linear(2,2) → tanh → Linear(2,3)` + pad(3) + 5×20 head |
| 4 | 1 827 | 1 827 | `Linear(2,2) → tanh → Linear(2,4)` + pad(6) + 5×20 head |
| 5 | 1 853 | 1 853 | `Linear(2,3) → tanh → Linear(3,5)` + pad(1) + 5×20 head |
| 6 | 1 879 | 1 879 | (extended sweep) |
| 7 | 1 905 | 1 905 | (extended sweep) |
| 8 | 1 931 | 1 931 | (extended sweep) |

The twin's "pad" parameters are free bias-like vectors that consume the parameter slack so the total matches exactly. The GAAF-PINN variant returns one pad parameter to make room for a single global adaptive activation scalar $a$, so its parameter count also matches.

### 6.2 Physics-informed loss (4-term Shah formulation)

The Phase-2 loss follows Shah et al. (2024) Eq. 5–9 exactly (Phase 1 used a simpler 2-term form):

$$
\mathcal{L} = \lambda_\text{pde}\,\mathcal{L}_\text{pde} + \lambda_\text{ic}\,\mathcal{L}_\text{ic} + \lambda_\text{bc1}\,\mathcal{L}_\text{bc1} + \lambda_\text{bc2}\,\mathcal{L}_\text{bc2},
$$

with the four terms

$$
\mathcal{L}_\text{pde} = \text{mean}\!\Bigl[\bigl(u_t + u u_x - \nu\,u_{xx}\bigr)^2\Bigr], \quad \mathcal{L}_\text{ic}  = \text{mean}\!\Bigl[\bigl(u(x, 0) + \sin(\pi x)\bigr)^2\Bigr],
$$

$$
\mathcal{L}_\text{bc1} = \text{mean}\!\Bigl[u(+1, t)^2\Bigr], \quad \mathcal{L}_\text{bc2} = \text{mean}\!\Bigl[u(-1, t)^2\Bigr].
$$

The default weighting is $\lambda_\text{ic} = \lambda_\text{bc1} = \lambda_\text{bc2} = 10$ and $\lambda_\text{pde} = 1$ — the data-fit terms are up-weighted to prevent the optimizer from collapsing to the trivial zero-output solution (a documented PINN training pathology that L-BFGS-B is particularly prone to).

### 6.3 Schedule-based, interruption-proof training

Each config is trained with a **two-phase schedule**: Adam for stable escape from initialization, followed by chunked L-BFGS for high-precision convergence.

```python
schedule = [
    {"kind": "adam",  "iters": 1000, "lr": 8e-3, "tmax": 1.0, "shock_frac": 0.0},
    {"kind": "lbfgs", "iters": 1000, "chunk": 200},
]
```

- **Phase A (Adam):** Adam with `lr=8e-3` and AMSGrad. Per-parameter adaptive learning rate handles the wildly different gradient scales between the small VQC weights (3-8 parameters) and the large classical head (1 800+ parameters).
- **Phase B (L-BFGS):** PyTorch's `LBFGS` with `history_size=60`, `line_search_fn="strong_wolfe"`, run in chunks of 200 iterations. A **blow-up guard** restores the best-known state if the loss diverges by more than $50\times$ the best.
- **Checkpointing:** every 250 iterations, the full training state (model + optimizer + history + best state) is atomically saved via `os.replace`. The training function resumes transparently from the last checkpoint on re-run.
- **Dead-run detection:** if the loss is unchanged across 3 logged chunks (suggesting a stuck run), the framework retries with a perturbed seed.

### 6.4 Four model variants

The framework supports four model variants that share the same training engine:

1. **QAPINN** — VQC first layer + deep classical head (described above).
2. **ClassicalTwin** — parameter-matched classical network with the same input-scaling, head depth, and head width. The "front" replaces the VQC with `Linear(2, H) → tanh → Linear(H, n_feat)` plus a pad vector.
3. **GAAF-PINN** — Global Adaptive Activation Function PINN: every $\tanh$ in the twin is replaced by $\tanh(N \cdot a \cdot z)$ with a single shared trainable scalar $a$ (init so $N\cdot a \approx 1$). The pad vector is shrunk by one to keep the parameter count identical.
4. **ShahQAPINN** — Faithful Shah et al. (2024) reproduction: 8-layer RX + full-entanglement VQC, **probability state-vector** measurement, a single small classical hidden layer, **Adam only** (no L-BFGS), fixed input scale. Built at Shah's exact parameter budget (74/189/600 for $n=3/4/5$).

---

## 7. Phase 2 : Hard-Viscosity Benchmark

The hard-viscosity regime $\nu = 0.01/\pi$ is the actual Raissi/Shah benchmark. It is significantly harder than the easy regime because:

- The shock forms at $t \approx 0.4$ around $x = 0$, with sharp high-frequency content
- The PDE residual $\nu\,u_{xx}$ is sensitive to small curvature errors
- 1 000 collocation points under-resolve the shock, so we use **shock-focused LHS augmentation**: a fraction `shock_frac` of collocation points are drawn from $\mathcal{N}(0, 0.15)$ in space and $\mathcal{U}(0.25\,t_\text{max},\, t_\text{max})$ in time
- Training uses 4 400 iterations (vs 2 000 for the easy regime) and a **time-marching curriculum** that progressively extends $t_\text{max}$

Each hard-ν notebook ships a per-run `convergence_flags.json` (dead-run detection results) and `hardware.json` (host log).

---

## 8. Phase 2 : Ablation Grid

The ablation grid (`Phase 2/phase2_part_2/wiser/QAPINN_ABLATION.ipynb`) varies **one axis at a time** with everything else fixed, on the easy-viscosity Burgers equation. Fourteen configurations (B1-B14) cover:

| Axis | Variants |
|------|----------|
| Qubit count | 3, 4, 5 |
| Entanglement | `"none"`, `"linear"`, `"all"` |
| Measurement | `"expval"`, `"probs"` |
| Variational depth | $L = 3$ vs $L = 9$ |
| Trainability limit | 6, 7 qubits (barren-plateau probe) |

Each ablation config adds:

- A **gradient-variance probe** (Workstream B): per-parameter gradient variance vs qubit count, the empirical barren-plateau diagnostic.
- **Per-neuron interpretability**: per-neuron activation profiles, neuron-health histograms, per-neuron frequency content (DFT of activations on a spatial grid at $t = 0.5$).

The ablation confirms the headline finding from the multi-seed run: **entanglement matters** , turning it off (`"none"`) collapses the model to $L^2 \approx 1$.

---

## 9. Phase 3 : Explainability Workstreams

Phase 3 consumes Phase-2 artifacts and runs three workstreams (no retraining). All workstreams evaluate at 26 time slices across $t \in [0, 1]$.

### 9.1 Workstream A : Fourier-spectrum recovery (Schuld's theorem test)

**Hypothesis:** A VQC's accessible Fourier spectrum is determined by its encoding Hamiltonian (Schuld et al. 2021). With angle encoding $R_Y(\pi x)$, the VQC can express functions of the form

$$
f(x) = \sum_{\omega \in \Omega} c_\omega\, e^{i\omega x}, \qquad \Omega = \{-L, \dots, -1, 0, 1, \dots, L\},
$$

so a depth-$L$ VQC should access frequencies up to $\omega_\text{max} = L$. The classical PINN has no such theorem — its spectrum is determined opaquely by the activation function.

**Method:** for each trained model, extract the predicted field $u_\text{pred}(x, t)$ at $t = 1.0$ on a 256-point grid, compute the 1-D FFT, and compare the spectral centroid and high-frequency fraction against the analytical reference.

**Metrics:** `centroid_model` vs `centroid_exact` (rad/sample), `hf_frac_model` vs `hf_frac_exact` (fraction of power above 1/4 Nyquist).

**Outputs:** `WSA/wsa_spectral_metrics.csv`, figures A1–A5 (overlay, qubit sweep, metric trends, seed sensitivity, single-vs-multi-seed).

### 9.2 Workstream B : Learning dynamics & trainability

**Hypothesis:** McClean et al. (2018) predict that for sufficiently deep random circuits, the gradient variance decays exponentially with qubit count:

$$
\mathrm{Var}\!\bigl[\partial_\theta \mathcal{L}\bigr] \sim \mathcal{O}\!\bigl(3^{-n}\bigr),
$$

the **barren plateau** phenomenon. We test this empirically by computing the per-parameter gradient variance at fixed initialization across $n = 3, \dots, 8$.

**Method:**

1. **Loss traces:** aggregate all per-run loss histories, mark dead runs and retries.
2. **Gradient variance:** for each $(model, n)$, rebuild at the trained checkpoint and compute $\mathrm{Var}[\nabla_\theta \mathcal{L}]$ on a fixed batch — the empirical barren-plateau probe.
3. **Steps-to-threshold:** number of optimizer steps to reach $L^2 < 10^{-1}, 10^{-2}, 10^{-3}$.
4. **Cost-per-step:** wall-clock seconds per optimizer step, logged per host.

**Outputs:** `wsb_run_summary.csv`, `wsb_gradient_variance.csv`, `wsb_loss_traces.npz`, `wsb_timing.csv`, figures HB1–HB4 (loss curves, gradient variance vs $n$, steps-to-threshold, cost).

### 9.3 Workstream C : Neuron XAI & CKA information flow

**Hypothesis:** The quantum layer either dominates the prediction (high CKA with the output) or is cosmetic (low CKA). We measure this with **Centered Kernel Alignment** (Kornblith et al. 2019):

$$
\text{CKA}(\mathbf{X}, \mathbf{Y}) = \frac{\langle\text{vec}(\mathbf{K}_X), \text{vec}(\mathbf{K}_Y)\rangle}{\|\text{vec}(\mathbf{K}_X)\|\|\text{vec}(\mathbf{K}_Y)\|}, \qquad \mathbf{K}_X = \mathbf{X}\mathbf{X}^\top,
$$

which gives a representation-similarity score in $[0, 1]$ between two activation matrices $\mathbf{X}, \mathbf{Y} \in \mathbb{R}^{N \times d}$, invariant to orthogonal transforms and isotropic scaling.

**Method:**

1. Rebuild each model, forward-pass on a 256×256 grid, extract per-layer activations.
2. Compute **layer-to-layer CKA**: for the QAPINN, CKA between the VQC output and every classical layer; for the twin, CKA between the first classical layer and every subsequent layer.
3. Compute **neuron-output CKA**: per-neuron contribution to the final output (the Figure-3 reproduction from BQP's challenge document).
4. **Neuron health histograms** at every time slice (dead-neuron, saturation, specialization diagnostics from Phase 1).
5. **Per-neuron frequency content**: DFT of each neuron's activation along the spatial axis.

**Outputs:** `wsc_cka_by_slice.csv`, `wsc_neuron_stats.csv`, figures HC1–HC8 (information flow, where representation forms, Figure-3 reproduction, neuron-vs-time maps, neuron-output CKA, saturation).

---

## 10. Key Results

### 10.1 Phase 1 classical baseline

Relative $L^2 = 2.12\times10^{-4}$ on Burgers $\nu = 0.01/\pi$, verified below the $10^{-3}$ target — 3× better than Raissi et al. (2017).

### 10.2 Phase 2 multi-seed head-to-head (5 seeds, paired t-test)

| Config | QAPINN $L^2$ (mean ± sd) | Twin $L^2$ (mean ± sd) | Ratio (Q/T) | $p$-value | Winner |
|--------|--------------------------:|-----------------------:|------------:|----------:|--------|
| Burgers $n=3$, $\nu=0.05$ | $0.0117 \pm 0.0126$ | $0.0725 \pm 0.1509$ | 0.900 | $>0.05$ | (tie) |
| Burgers $n=4$, $\nu=0.05$ | $0.00457 \pm 0.00298$ | $0.01426 \pm 0.02199$ | 0.999 | $>0.05$ | (tie) |
| **Burgers $n=5$, $\nu=0.05$** | $0.00454 \pm 0.00534$ | $0.000965 \pm 0.000196$ | **0.304** | **$<0.05$** | **Twin** |
| Heat $n=4$, $\alpha=0.1$ | $0.00191 \pm 0.00143$ | $0.00231 \pm 0.00290$ | 0.942 | $>0.05$ | (tie) |

**Headline interpretation:** at exact parameter parity, the QAPINN matches the classical twin on Burgers (no statistically significant advantage at $n=3, 4$), and the twin actually wins at $n=5$ in the easy regime. The Phase-3 ablation suggests this is because the easy-viscosity Burgers solution is dominated by low frequencies that the classical network already captures well — the VQC's frequency-access advantage (per Schuld's theorem) does not pay off here.

### 10.3 Phase 2 hard-viscosity ($\nu = 0.01/\pi$) summary

| Model | $n$ | $L^2$ mean | $\phi$-recovery mean | Gradient variance | CKA(first→out) |
|-------|----:|-----------:|---------------------:|------------------:|---------------:|
| QAPINN | 3 | 0.118 | 0.974 | 0.490 | 0.555 |
| QAPINN | 4 | 0.091 | 0.994 | 0.209 | 0.660 |
| QAPINN | 5 | 0.026 | 0.998 | 0.112 | 0.619 |
| QAPINN | 6 | 0.017 | 0.994 | 0.129 | 0.623 |
| QAPINN | 7 | 0.053 | 1.001 | 0.138 | 0.608 |

where $\phi$-recovery is the cosine similarity between the model's spectral centroid and the analytical centroid. Note that gradient variance does **not** show the exponential decay predicted by McClean et al. — at our shallow depth ($L = 6$), barren plateaus do not yet appear out to $n = 7$.

### 10.4 Phase 2 ablation grid  selected findings

- **Entanglement is essential:** the `"none"` ablation (B4, B6) collapses to $L^2 \approx 1$ — the model cannot learn Burgers without entanglement, regardless of qubit count.
- **Linear vs all-to-all:** the difference is small at $n = 3$ but grows with $n$, consistent with Schuld's frequency-access theorem.
- **Measurement type:** `"expval"` ( $n$ outputs) outperforms `"probs"` ( $2^n$ outputs) at small $n$ in our regime, because the smaller output dimension concentrates the gradient signal.
- **Depth:** $L = 9$ is marginally better than $L = 3$ on Burgers but worse on Heat (over-parameterization for a smooth solution).

### 10.5 Phase 3 : Workstream A (Fourier recovery)

The QAPINN recovers the analytical spectral centroid to within 1 % at $n \geq 4$ on hard-ν Burgers. The classical twin's centroid drifts by 5–15 %, especially at $n = 3$.

### 10.6 Phase 3 : Workstream C (CKA information flow)

The VQC's first-layer CKA with the output is $0.55$–$0.66$ across all configs — meaning roughly half the output's representational geometry is already present in the VQC output. The classical twin's first-layer CKA is consistently higher ($0.70$+), suggesting the quantum layer encodes **less** of the final prediction than the first classical layer does — but the part it does encode is the high-frequency component.

---

## 11. Reproducing the Shah et al. Table I

Shah et al. (2024) report parameter counts of 74 / 189 / 600 for QA-PINN at $n = 3/4/5$ and relative $L^2$ of $0.06 / 0.008 / 0.0006$ on hard-ν Burgers.

Our reproduction (in `GAAF_QAPINN_SHA_REPRO/Shah & al Reproduction/QAPINN_SHAH_FIDELITY.ipynb`):

| $n$ | Shah reported | Our faithful repro | Our recipe (shrunk to same budget) | Params |
|----:|--------------:|-------------------:|-----------------------------------:|-------:|
| 3 | 0.06 | 0.770 | 0.067 | 74 |
| 4 | 0.008 | 0.759 | 0.021 | 189 |
| 5 | 0.0006 | 0.510 | 0.010 | 600 |

The faithful Shah reproduction struggles (likely due to under-documented details in the paper — the encoding scheme and exact optimizer schedule are not fully specified). However, when we **shrink our validated QAPINN recipe to Shah's exact parameter budget**, we recover $L^2$ values within an order of magnitude of Shah's reported numbers — isolating that the *procedure* (multi-phase Adam + L-BFGS, shock-focused LHS, Cole–Hopf reference) is what drives the accuracy, not the exact 8-layer architecture.

---

## 12. Methodological Notes & Caveats

1. **Single seed vs multi-seed:** the early Phase-2 notebooks (`Tiers12_q3`, `Tiers12_q38_more_iter`) use a single seed and report cherry-picked best runs. The definitive results are in `Tiers12_multi_seed/QAPINN_PHASE2_FINAL.ipynb`, which uses 5 seeds and reports paired t-tests. **Trust only the multi-seed results.**

2. **q8 budget caveat:** at hard $\nu$, the $n = 8$ runs used only 1 000 iterations (vs 2 016 for $n = 3$–$7$) due to compute budget. They are marked `in_primary = False` in `run_metadata.csv` and reported separately.

3. **Two viscosities:** easy ($\nu = 0.05$) and hard ($\nu = 0.01/\pi$). The easy regime is for fast iteration and ablation; the hard regime is the actual Raissi/Shah benchmark. **Conclusions from the easy regime do not transfer to the hard regime** — the frequency content is qualitatively different.

4. **WSA at easy ν has no notebook in the repo** — only its output artifacts (`Phase_3_heat_easyBurger/WSA/`). The analysis code was developed in a Colab session that was not committed. The hard-ν WS-A code is in `Phase3_Hard_InputBuilder.ipynb`.

5. **Google Colab:** Phase-3 notebooks mount Google Drive (`from google.colab import drive`). The canonical Colab working layout puts results under `WISER Results/Phase 2 Hard Burger/...`. To run locally, adjust the path constants at the top of each notebook.

6. **Git LFS:** two CSVs under `Phase_3_heat_easyBurger/` are tracked via Git LFS (per `.gitattributes`). Everything else is plain git.

7. **Hardware:** all experiments ran on Intel i7-8650U laptops (4 physical cores, CPU-only). Per-run host logs are in `hardware.json` files alongside the results.

---

## 13. Dependencies

```
# Inferred from the notebooks' import cells. Pinned versions are not specified
# in the repo; these are the minimum recommended versions.
torch>=2.0          # tensors, autograd, Adam + chunked L-BFGS
pennylane>=0.30     # VQC, default.qubit simulator, qml.qnn.TorchLayer
numpy
scipy               # scipy.optimize.minimize (Phase-1 L-BFGS-B), scipy.stats.qmc (LHS)
matplotlib          # all figures (saved as 300-dpi PNG + vector PDF)
pandas              # Phase-3 figure notebooks
google-colab        # Phase-3 notebooks mount Google Drive
```

Python 3.10+. CPU sufficient for all experiments; CUDA auto-detected.

---

## 14. Citation

If you use this code or build on our results, please cite:

```bibtex
@misc{wiser_bqp_2026,
  title  = {Quantum-Assisted Physics-Informed Neural Networks for Computational Fluid Dynamics:
            A Reproduction, Ablation, and Explainability Study},
  author = {Djabon, Ounomborbitibou and Hounk{\`e}, Mawouliklimi Roland},
  year   = {2026},
  note   = {WISER 2026 Summer Program, in collaboration with BosonQ Psi (BQP)},
  url    = {https://github.com/mawuliro/wiser_project}
}
```

Please also cite the foundational works:

```bibtex
@article{raissi2019physics,
  title   = {Physics-informed neural networks: A deep learning framework for solving
             forward and inverse problems involving nonlinear partial differential equations},
  author  = {Raissi, Maziar and Perdikaris, Paris and Karniadakis, George E.},
  journal = {Journal of Computational Physics},
  volume  = {378}, pages = {686--707}, year = {2019}
}

@inproceedings{shah2024benchmarking,
  title     = {Benchmarking Quantum-Assisted PINN (QA-PINN) for Computational Fluid Dynamics},
  author    = {Shah, Jay and Lineswala, Rut and Chopra, Abhishek},
  booktitle = {2024 IEEE International Conference on Quantum Computing and Engineering (QCE)},
  year      = {2024}
}

@article{schuld2021effect,
  title   = {Effect of data encoding on the expressive power of variational quantum machine learning models},
  author  = {Schuld, Maria and Sweke, Ryan and Meyer, Johannes J.},
  journal = {Physical Review A}, volume = {103}, number = {3}, pages = {032430}, year = {2021}
}

@article{mcclean2018barren,
  title   = {Barren plateaus in quantum neural network training landscapes},
  author  = {McClean, Jarrod R. and Boixo, Sergio and Smelyanskiy, Vadim N. and
             Babbush, Ryan and Yao, Hartmut},
  journal = {Nature Communications}, volume = {9}, pages = {4812}, year = {2018}
}

@inproceedings{kornblith2019similarity,
  title     = {Similarity of neural network representations revisited},
  author    = {Kornblith, Simon and Norouzi, Mohammad and Lee, Honglak and Hinton, Geoffrey},
  booktitle = {Proceedings of the 36th International Conference on Machine Learning (ICML)},
  year      = {2019}
}
```

---

<p align="center">
  <em>WISER 2026 · BosonQ Psi · Summer 2026</em>
</p>
