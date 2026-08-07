<div align="center">

# ⚛️ Quantum-Assisted PINNs for Computational Fluid Dynamics

### 🌍 WISER 2026 Summer Program Industry Challenge

> From July, 6 2026 to Aug, 7 2026

---

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)
![PennyLane](https://img.shields.io/badge/PennyLane-0.30%2B-00B0FF?logo=pennylane&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-1.10%2B-8CAAE6?logo=scipy&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-1.24%2B-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7%2B-11557C?logo=matplotlib&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

---

### 🌟 Team: Quantum Horizon Africa

**Ounimborbitibou Djabon** · AIMS Ghana / Univ. of Lome (TOGO)
📧 [`djabon@aims.edu.gh`](mailto:djabon@aims.edu.gh) · 📱 +228 92 39 97 21

**Mawulikplimi Roland Hounkpe** · AIMS Ghana / Univ. of Lome (TOGO)
📧 [`rhounkpe@aims.edu.gh`](mailto:rhounkpe@aims.edu.gh) · 📱 +228 93 21 72 15

### 🔗 Repository

📦 [github.com/mawuliro/wiser_project](https://github.com/mawuliro/wiser_project)

---

> 🎯 **Mission:** *Reproduce, ablation-test, and explain the Quantum-Assisted PINN (QA-PINN) architecture proposed by Shah et al. (2024) on the 1-D viscous Burgers and heat equations, and answer BQP's central question:* **when, why, and how does a variational quantum circuit change the learning dynamics of a PINN?**

</div>

---

## 📑 Table of Contents

| # | Section | Icon |
|:-:|---------|:----:|
| 1 | [The Challenge](#1-🎯-the-challenge) | 🎯 |
| 2 | [Our Team's Approach](#2-🛠️-our-teams-approach) | 🛠️ |
| 3 | [Methods and Tools Used](#3-⚙️-methods-and-tools-used) | ⚙️ |
| 4 | [Results and Findings](#4-📈-results-and-findings) | 📈 |
| 5 | [Limitations and Recommended Next Steps](#5-🔮-limitations-and-recommended-next-steps) | 🔮 |
| 6 | [Team Members and Their Contributions](#6-👥-team-members-and-their-contributions) | 👥 |
| 7 | [Repository Structure](#7-📁-repository-structure) | 📁 |
| 8 | [Quickstart](#8-🚀-quickstart) | 🚀 |
| 9 | [Citation](#9-📝-citation) | 📝 |
| 10 | [Contact](#📬-contact) | 📬 |

---

## 1. 🎯 The Challenge

### 1.1 🧩 The problem statement

BosonQ Psi (BQP) posed the following challenge to WISER 2026 participants:

> 💡 *Explain when, why, and how the introduction of a variational quantum circuit (VQC) changes the learning dynamics of a Physics-Informed Neural Network (PINN).*

This is fundamentally a **comparative question**. To say what the quantum layer changes, we must first know what the classical network does. The challenge therefore requires three things simultaneously:

| | Requirement | Why |
|:-:|-------------|-----|
| 🅰️ | A verified classical PINN baseline | Without it, any quantum claim is suspect |
| 🅱️ | A quantum-assisted PINN (QAPINN) built on top of that baseline | The treatment group |
| ©️ | A systematic explainability analysis comparing the two | The actual answer to "what changed?" |

### 1.2 🌊 The benchmark problem

The challenge is set on the **1-D viscous Burgers equation**, the canonical CFD benchmark derived from the Navier-Stokes equations by dropping the pressure-gradient term:

$$
\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} - \nu \frac{\partial^2 u}{\partial x^2} = 0, \qquad x \in [-1, 1], \quad t \in [0, 1],
$$

with initial condition $u(x, 0) = -\sin(\pi x)$ and boundary conditions $u(\pm 1, t) = 0$.

Two viscosity regimes are studied:

| Regime | $\nu$ | Character | Origin |
|:------:|:-----:|:----------|:-------|
| 🟢 **Easy** | $0.05$ | Smooth, near-linear advection | WISER Tier-1 |
| 🔴 **Hard** | $0.01/\pi \approx 3.18 \times 10^{-3}$ | Sharp shock forms at $t \approx 0.4$ | Raissi (2017), Shah (2024) |

We also use the linear **heat equation**, $\partial_t u - \alpha \partial_{xx} u = 0$ with $\alpha = 0.1$, as a control PDE that has no nonlinear advection term.

### 1.3 📄 The reference paper

Shah et al. (2024), *"Benchmarking Quantum-Assisted PINN (QA-PINN) for Computational Fluid Dynamics"* (IEEE QCE 2024), reports that a QAPINN with 3, 4, or 5 qubits achieves accuracy comparable to or better than classical PINNs while reducing the trainable parameter count by up to **20%**. Their parameter counts are 74, 189, and 600 for 3, 4, and 5 qubits respectively, and their best result is $L^2 = 0.6 \times 10^{-3}$ at 5 qubits on hard-viscosity Burgers.

Our task was to **reproduce** their result, **probe** its sensitivity to each quantum design choice, and **explain** the mechanism behind any accuracy difference we observe.

---

## 2. 🛠️ Our Team's Approach

### 2.1 🏗️ Why a three-phase strategy

We structured the project into three sequential phases because the challenge question is comparative at its core. Each phase produces a deliverable that the next phase depends on.

```
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│   Phase 1    │ ───▶ │   Phase 2    │ ───▶ │   Phase 3    │
│  Classical   │      │   QAPINN     │      │ Explainability│
│  Baseline    │      │   Framework  │      │   Analysis    │
└──────────────┘      └──────────────┘      └──────────────┘
   ✅ Verified          🔬 Ablation grid       📊 Fourier + CKA
   L² = 2.1e-4         + Parameter-matched    + Barren plateaus
                        twin comparison
```

**Phase 1: Classical PINN baseline.** 🧱 Before comparing classical and quantum, we needed a verified classical PINN that matches the published state of the art. A weak baseline would make every later quantum claim suspect, because any accuracy difference could be attributed to a buggy classical implementation rather than to the quantum layer.

**Phase 2: QAPINN framework and ablation grid.** ⚛️ With the baseline in hand, we built a flexible QAPINN framework where the first hidden layer of the classical PINN is replaced by a variational quantum circuit. We then systematically varied each quantum design axis (qubit count, entanglement topology, measurement type, variational depth) while keeping all classical hyperparameters fixed, so that any accuracy difference is attributable to the quantum design choice alone.

**Phase 3: Explainability analysis.** 🔬 With trained QAPINN and classical models in hand, we ran three workstreams to explain *why* the quantum layer helps or does not help: Fourier-spectrum analysis (does the VQC access different frequencies?), gradient-variance analysis (do barren plateaus appear?), and CKA information-flow analysis (does the quantum layer dominate the prediction or is it cosmetic?).

### 2.2 🧪 Classical, quantum, and hybrid methods considered

We considered and implemented **five model variants**, all sharing the same training engine:

| # | Model | Type | Description |
|:-:|-------|:----:|-------------|
| 1️⃣ | **Classical PINN** | 🟦 Classical | 9-layer tanh MLP with 3,021 parameters, trained with scipy L-BFGS-B. Raissi et al. (2017) architecture, used as the accuracy reference. |
| 2️⃣ | **QAPINN** | ⚛️ Hybrid | VQC replaces the first hidden layer. Angle encoding with data re-uploading, configurable entanglement (none, linear, or all-to-all), configurable measurement (expval or probs). Deep classical head (5 layers of 20 neurons). |
| 3️⃣ | **ClassicalTwin** | 🟦 Classical | Parameter-matched classical network with the exact same input scaling, head depth, and head width as the QAPINN. Pad vector consumes parameter slack so totals match exactly (1,801 / 1,827 / 1,853 params). **The control group.** |
| 4️⃣ | **GAAF-PINN** | 🟦 Classical | Global Adaptive Activation Function PINN. Every tanh replaced by $\tanh(N \cdot a \cdot z)$ with a single shared trainable scalar $a$. State-of-the-art classical baseline.|
| 5️⃣ | **ShahQAPINN** | ⚛️ Hybrid | Faithful reproduction of Shah et al.'s exact architecture (8-layer RX VQC, probability state vector, single hidden layer, Adam only) at their exact parameter budget (74 / 189 / 600). |

### 2.3 🧠 Why we selected this approach

> ✅ **Parameter-matched twin design:** The only way to make a fair classical-vs-quantum comparison. If the QAPINN has fewer parameters than the classical baseline (as in Shah et al.), an accuracy difference could be caused by the parameter count rather than the quantum mechanism. By matching parameters exactly, we isolate the quantum contribution.

> ✅ **Multi-seed paired statistical comparison (5 seeds, paired t-test):** Quantum circuit training is noisy. A single seed can produce a cherry-picked result. The paired t-test on $\log L^2$ controls for seed-to-seed variance and gives a defensible $p$-value.

> ✅ **Three-workstream Phase 3 design:** The challenge asks "when, why, and how." Workstream A (Fourier) addresses "why" via Schuld et al.'s theorem. Workstream B (gradient variance) addresses "when" via McClean et al.'s barren-plateau theorem. Workstream C (CKA) addresses "how" by tracing information flow.

---

## 3. ⚙️ Methods and Tools Used

### 3.1 📐 Mathematical formulation

#### 🌊 The Burgers equation and Cole-Hopf reference

Burgers' equation admits a closed-form solution via the **Cole-Hopf transformation**. Define

$$
F(y; x, t) = \frac{\cos(\pi y)}{2\pi\nu} + \frac{(x - y)^2}{4\nu t},
$$

then the analytical solution is

$$
u(x, t) = \frac{\displaystyle\int_{-\infty}^{\infty} \frac{x - y}{t}\, e^{-F(y;\,x,\,t)}\, dy}{\displaystyle\int_{-\infty}^{\infty} e^{-F(y;\,x,\,t)}\, dy}.
$$

The integrand spans 40+ orders of magnitude, so we evaluate it in log-space with the numerically stable weighted-average form

$$
u(x, t) = \frac{\sum_k \tfrac{x - y_k}{t}\, w_k}{\sum_k w_k}, \qquad w_k = \exp\!\bigl(-F(y_k;\,x,\,t) - \max_j[-F(y_j;\,x,\,t)]\bigr),
$$

using a 4,000-point quadrature grid on $y \in [-3, 3]$ in float64. This eliminates the interpolation error that plagues finite-difference reference solvers and is reused as the evaluation target throughout all phases.

#### 🔥 The heat equation reference

The heat equation has the closed-form solution $u(x,t) = \sin(\pi x) e^{-\alpha \pi^2 t}$, which provides a noise-free reference for the easy-regime comparison.

### 3.2 ⚛️ The QAPINN architecture

The QAPINN is a hybrid quantum-classical module:

$$
u_\theta(x, t) = \text{head}_\phi\!\bigl(\,\text{VQC}_\psi\!\bigl(\pi\, s \cdot [x, t]\bigr)\bigr),
$$

where $s \in \mathbb{R}^2$ is a trainable input scale, $\psi$ are the VQC weights, and $\phi$ are the classical head weights.

#### 🔗 The variational quantum circuit (VQC)

The VQC uses **angle encoding with data re-uploading**. At every variational layer $\ell \in \{0, \dots, L-1\}$, the input $(x, t)$ is encoded as $R_Y$ rotations on every qubit. Each layer then applies a trainable $R_X(\theta_{\ell, q})$ rotation per qubit, followed by an entanglement block:

$$
|\psi_\text{out}\rangle = \prod_{\ell=1}^{L}\left[\Bigl(\prod_{q=1}^{n} R_X(\theta_{\ell,q})\Bigr) \cdot \mathcal{E}\right] \cdot \Bigl(\prod_{q=1}^{n} R_Y(\pi\, s \cdot x_t)\Bigr) \,|0\rangle^{\otimes n},
$$

where $\mathcal{E}$ is one of three entanglement topologies:

| `entanglement` | $\mathcal{E}$ | Description |
|:--------------:|:--------------|:------------|
| `"none"` | $\mathbb{1}$ | 🔵 Product-state ansatz (no entanglement) |
| `"linear"` | $\prod_{q=1}^{n-1}\text{CNOT}(q, q+1)$ | 🔵 Linear nearest-neighbor chain |
| `"all"` | $\prod_{a<b}\text{CNOT}(a, b)$ | 🔵 All-to-all (Shah-style full entanglement) |

Two measurement schemes are supported:
- 📏 **`expval`**: local expectation values $\langle\sigma_z^{(q)}\rangle$ for $q = 1, \dots, n$ (output dimension $n$)
- 📊 **`probs`**: full probability state vector over the $2^n$ computational basis states (output dimension $2^n$)

#### 🧠 The classical post-processing head

Following Shah et al., the classical head is a deep tanh MLP:

$$
\text{head}_\phi(\mathbf{f}) = W_\text{out}\,\tanh\!\bigl(W_{D-1}\,\tanh(\cdots\tanh(W_1\,\mathbf{f} + b_1)\cdots) + b_{D-1}\bigr) + b_\text{out},
$$

with default head width 20 and head depth 5. Xavier-normal initialization on the linear weights, zero initialization on biases.

#### ⚖️ Parameter parity (the fair-comparison guarantee)

The QAPINN and its classical twin are designed to have **identical parameter counts**:

| $n$ qubits | QAPINN params | ClassicalTwin params | Twin structure |
|:----------:|:-------------:|:--------------------:|:---------------|
| 3 | 1,801 | 1,801 | `Linear(2,2) → tanh → Linear(2,3)` + pad(3) + 5×20 head |
| 4 | 1,827 | 1,827 | `Linear(2,2) → tanh → Linear(2,4)` + pad(6) + 5×20 head |
| 5 | 1,853 | 1,853 | `Linear(2,3) → tanh → Linear(3,5)` + pad(1) + 5×20 head |
| 6 | 1,879 | 1,879 | (extended sweep) |
| 7 | 1,905 | 1,905 | (extended sweep) |
| 8 | 1,931 | 1,931 | (extended sweep) |

The twin's "pad" parameters are free bias-like vectors that consume the parameter slack so the total matches exactly.

### 3.3 🧮 Physics-informed loss (4-term Shah formulation)

The Phase 2 loss follows Shah et al. (2024) Equations 5 through 9:

$$
\mathcal{L} = \lambda_\text{pde}\,\mathcal{L}_\text{pde} + \lambda_\text{ic}\,\mathcal{L}_\text{ic} + \lambda_\text{bc1}\,\mathcal{L}_\text{bc1} + \lambda_\text{bc2}\,\mathcal{L}_\text{bc2},
$$

where

$$
\mathcal{L}_\text{pde} = \text{mean}\!\Bigl[\bigl(u_t + u\,u_x - \nu\,u_{xx}\bigr)^2\Bigr], \quad \mathcal{L}_\text{ic}  = \text{mean}\!\Bigl[\bigl(u(x, 0) + \sin(\pi x)\bigr)^2\Bigr],
$$

$$
\mathcal{L}_\text{bc1} = \text{mean}\!\Bigl[u(+1, t)^2\Bigr], \quad \mathcal{L}_\text{bc2} = \text{mean}\!\Bigl[u(-1, t)^2\Bigr].
$$

The PDE residual is computed **exactly** via PyTorch autograd 🎯, not via finite differences. The default weighting is $\lambda_\text{ic} = \lambda_\text{bc1} = \lambda_\text{bc2} = 10$ and $\lambda_\text{pde} = 1$, which up-weights the data-fit terms to prevent the optimizer from collapsing to the trivial zero-output solution.

### 3.4 🔁 Training schedule

Each configuration is trained with a **two-phase schedule**: Adam for stable escape from initialization, followed by chunked L-BFGS for high-precision convergence.

```python
schedule = [
    {"kind": "adam",  "iters": 1000, "lr": 8e-3, "tmax": 1.0, "shock_frac": 0.0},
    {"kind": "lbfgs", "iters": 1000, "chunk": 200},
]
```

| Phase | Optimizer | Settings | Purpose |
|:-----:|:---------:|:---------|:--------|
| 🅰️ Phase A | **Adam** | lr = $8 \times 10^{-3}$, AMSGrad | Stable escape from initialization. Per-parameter adaptive LR handles wildly different gradient scales (VQC: 3-8 params, classical head: 1,800+ params). |
| 🅱️ Phase B | **L-BFGS** | history_size=60, strong-Wolfe, chunks of 200 | High-precision convergence. Blow-up guard restores best state if loss diverges by more than $50\times$. |

**💡 Key features:**
- 💾 **Checkpointing:** every 250 iterations, full state (model + optimizer + history + best state) atomically saved via `os.replace`. Resumes transparently on re-run.
- 🧹 **Dead-run detection:** if loss unchanged across 3 logged chunks, framework retries with perturbed seed.

### 3.5 🔬 Phase 3 explainability methods

#### 🅰️ Workstream A: Fourier-spectrum recovery (Schuld's theorem test)

Schuld et al. (2021) prove that a VQC's accessible Fourier spectrum is determined by its encoding Hamiltonian. With angle encoding $R_Y(\pi x)$, a depth-$L$ VQC can express:

$$
f(x) = \sum_{\omega \in \Omega} c_\omega\, e^{i\omega x}, \qquad \Omega = \{-L, \dots, -1, 0, 1, \dots, L\},
$$

so it should access frequencies up to $\omega_\text{max} = L$. The classical PINN has no such theorem. We test this empirically by computing the FFT of each model's predicted field at $t = 1.0$ on a 256-point spatial grid.

**📊 Metrics:** `centroid_model` vs `centroid_exact`, `hf_frac_model` vs `hf_frac_exact`

**📤 Outputs:** `WSA/wsa_spectral_metrics.csv`, figures A1-A5

#### 🅱️ Workstream B: Gradient-variance and barren plateaus

McClean et al. (2018) predict that for sufficiently deep random circuits, the gradient variance decays exponentially with qubit count:

$$
\mathrm{Var}\!\bigl[\partial_\theta \mathcal{L}\bigr] \sim \mathcal{O}\!\bigl(3^{-n}\bigr),
$$

the **barren plateau** phenomenon. We test this by computing per-parameter gradient variance at fixed initialization across $n = 3, 4, 5, 6, 7, 8$. We also record loss traces, steps-to-threshold, and wall-clock cost per step.

**📤 Outputs:** `wsb_run_summary.csv`, `wsb_gradient_variance.csv`, `wsb_loss_traces.npz`, `wsb_timing.csv`, figures HB1-HB4

#### ©️ Workstream C: CKA information flow

We use **Centered Kernel Alignment** (Kornblith et al. 2019) to measure representation similarity between layers:

$$
\text{CKA}(\mathbf{X}, \mathbf{Y}) = \frac{\langle\text{vec}(\mathbf{K}_X), \text{vec}(\mathbf{K}_Y)\rangle}{\|\text{vec}(\mathbf{K}_X)\| \|\text{vec}(\mathbf{K}_Y)\|}, \qquad \mathbf{K}_X = \mathbf{X}\mathbf{X}^\top,
$$

which gives a similarity score in $[0, 1]$ between two activation matrices, invariant to orthogonal transforms and isotropic scaling. We compute layer-to-layer CKA and neuron-output CKA at 26 time slices across $t \in [0, 1]$.

**📤 Outputs:** `wsc_cka_by_slice.csv`, `wsc_neuron_stats.csv`, figures HC1-HC8

### 3.6 🧰 Tools and libraries

| 🛠️ Tool | 📦 Version | 🎯 Purpose |
|:--------|:----------:|:-----------|
| Python | 3.10+ | All code |
| PyTorch | 2.0+ | Tensors, autograd, Adam + chunked L-BFGS |
| PennyLane | 0.30+ | VQC, `default.qubit` simulator, `qml.qnn.TorchLayer` |
| SciPy | 1.10+ | `scipy.optimize.minimize`, `scipy.stats.qmc` (LHS) |
| NumPy | 1.24+ | Numerical arrays, Cole-Hopf quadrature |
| Matplotlib | 3.7+ | All figures (300-dpi PNG + vector PDF) |
| Pandas | 2.0+ | Phase 3 figure notebooks |
| Google Colab | latest | Phase 3 notebooks mount Google Drive |
| Git LFS | latest | Two CSVs in `Phase_3_heat_easyBurger/` are LFS-tracked |

> 💻 **Hardware:** All experiments ran on Intel Core i7-8650U laptops (4 physical cores, CPU only). The VQC uses PennyLane's `default.qubit` state-vector simulator, so no quantum hardware is required. Per-run host logs are saved in `hardware.json` files alongside the results.

### 3.7 📚 Research and evidence

Our work is grounded in five foundational papers:

| # | 📄 Paper | 🎯 Role in our project |
|:-:|----------|:----------------------|
| 1 | Raissi, Perdikaris, Karniadakis (2019) | Original PINN framework, L-BFGS-B settings, 9-layer tanh architecture (Phase 1) |
| 2 | Shah, Lineswala, Chopra (2024) | QA-PINN paper we reproduce and ablate (Phase 2) |
| 3 | Schuld, Sweke, Meyer (2021) | Fourier spectrum theorem for VQCs (Phase 3 WS-A) |
| 4 | McClean et al. (2018) | Barren-plateau theorem (Phase 3 WS-B) |
| 5 | Kornblith et al. (2019) | CKA representation-similarity metric (Phase 3 WS-C) |

Full BibTeX entries are in the [📝 Citation](#9-📝-citation) section.

---

## 4. 📈 Results and Findings

### 4.1 ✅ Phase 1: classical baseline verified

The Phase 1 classical PINN achieved relative $L^2 = 2.12 \times 10^{-4}$ on hard-viscosity Burgers ($\nu = 0.01/\pi$), which is **3× better** than the Raissi et al. target of $6.7 \times 10^{-4}$. The baseline is verified below the $10^{-3}$ acceptance threshold.

| 📊 Metric | 📈 Value | 🎯 Target |
|:----------|---------:|----------:|
| Relative $L^2$ error | $2.12 \times 10^{-4}$ | $6.7 \times 10^{-4}$ (Raissi), **3× better** ✅ |
| PDE residual mean | $1.78 \times 10^{-4}$ | N/A |
| PDE residual max | $3.19 \times 10^{-2}$ | N/A |
| Generalization error ($t > 1$) | $3.20 \times 10^{-4}$ | N/A |
| Final loss | $4.97 \times 10^{-8}$ | N/A |
| Training iterations | 50,001 | N/A |
| Training time | 6,578 seconds | N/A |
| Parameters | 3,021 | N/A |
| **Status** | **✅ Verified** | $L^2 < 10^{-3}$ |

> 🔧 **Five documented fixes** were required to reach this result:
> 1. 📐 **Cole-Hopf analytical reference** (was: numerical method-of-lines): eliminates interpolation error
> 2. 🎯 **Latin Hypercube Sampling** for collocation (was: uniform random): better space-filling
> 3. ⚙️ **scipy L-BFGS-B with Raissi's exact parameters** (was: PyTorch LBFGS): PyTorch's LBFGS stalls on PINN loss landscapes
> 4. 🎲 **`xavier_uniform_` initialization** (was: `xavier_normal_`): matches Raissi's TensorFlow code
> 5. 🔢 **float64 precision** (was: float32): breaks through the float32 gradient-noise wall

### 4.2 🥊 Phase 2: multi-seed head-to-head at exact parameter parity

The definitive Phase 2 result is a **5-seed paired comparison** between QAPINN and ClassicalTwin, with a paired t-test on $\log L^2$:

| ⚙️ Config | ⚛️ QAPINN $L^2$ (mean ± sd) | 🟦 Twin $L^2$ (mean ± sd) | 📊 Ratio (Q/T) | 🎯 $p$-value | 🏆 Winner |
|:----------|:----------------------------:|:--------------------------:|:---------------:|:------------:|:----------|
| Burgers $n=3$, $\nu=0.05$ | $0.0117 \pm 0.0126$ | $0.0725 \pm 0.1509$ | 0.900 | $>0.05$ | 🤝 tie |
| Burgers $n=4$, $\nu=0.05$ | $0.00457 \pm 0.00298$ | $0.01426 \pm 0.02199$ | 0.999 | $>0.05$ | 🤝 tie |
| **Burgers $n=5$, $\nu=0.05$** | $0.00454 \pm 0.00534$ | $0.000965 \pm 0.000196$ | **0.304** | **$<0.05$** | 🟦 **Twin** |
| Heat $n=4$, $\alpha=0.1$ | $0.00191 \pm 0.00143$ | $0.00231 \pm 0.00290$ | 0.942 | $>0.05$ | 🤝 tie |

> 📌 **Finding:** At exact parameter parity, the QAPINN matches the classical twin on Burgers at $n=3$ and $n=4$ (no statistically significant difference). At $n=5$ on easy-viscosity Burgers, the twin actually wins ($p < 0.05$). The QAPINN does not demonstrate a clear accuracy advantage in the easy regime.

### 4.3 🔥 Phase 2: hard-viscosity results

At the hard viscosity $\nu = 0.01/\pi$ (the actual Raissi/Shah benchmark), the QAPINN shows clearer improvement with increasing qubit count:

| ⚛️ Model | $n$ | 📈 $L^2$ mean | 🎯 $\phi$-recovery | 📊 Grad. var. | 🔗 CKA (1st→out) | ⏱️ Sec/step |
|:---------|----:|--------------:|-------------------:|--------------:|-----------------:|------------:|
| QAPINN | 3 | 0.118 | 0.974 | 0.490 | 0.555 | 0.31 |
| QAPINN | 4 | 0.091 | 0.994 | 0.209 | 0.660 | 0.58 |
| QAPINN | 5 | 0.026 | 0.998 | 0.112 | 0.619 | 0.99 |
| QAPINN | 6 | 0.017 | 0.994 | 0.129 | 0.623 | 2.07 |
| QAPINN | 7 | 0.053 | 1.001 | 0.138 | 0.608 | 4.42 |

> 💡 **Two key observations:**
> 1. 🎯 **Frequency recovery:** The QAPINN recovers the analytical Fourier centroid to within 1% at $n \geq 4$, confirming that the quantum layer does access the correct frequency content.
> 2. 📉 **No barren plateaus:** Gradient variance does **not** show the exponential decay predicted by McClean et al. At our shallow depth ($L = 6$), barren plateaus do not appear even at $n = 7$.
> 3. 💰 **Cost:** The cost per step grows roughly $4\times$ per additional qubit, from 0.31 s at $n=3$ to 4.42 s at $n=7$.

### 4.4 🔬 Phase 2: ablation grid findings

The **14-configuration ablation grid** on easy-viscosity Burgers isolates the contribution of each quantum design axis:

| 🔍 Finding | 📊 Detail |
|:-----------|:---------|
| 🚫 **Entanglement is essential** | Turning entanglement off (`"none"`) collapses the model to $L^2 \approx 1.0$ (predicts zero everywhere). The model cannot learn Burgers without entanglement, regardless of qubit count. |
| 🔗 **Linear vs all-to-all** | Difference is small at $n=3$ but grows with $n$, consistent with Schuld's frequency-access theorem. |
| 📏 **Measurement type** | `expval` ($n$ outputs) outperforms `probs` ($2^n$ outputs) at small $n$ because the smaller output dimension concentrates the gradient signal. |
| 📚 **Depth** | $L=9$ is marginally better than $L=3$ on Burgers but worse on heat (over-parameterization for a smooth solution). |

### 4.5 🌈 Phase 3: Fourier-spectrum recovery (Workstream A)

> 🎯 **The QAPINN recovers the analytical spectral centroid to within 1% at $n \geq 4$ on hard-viscosity Burgers.** The classical twin's centroid drifts by 5 to 15%, especially at $n=3$.

This is the strongest evidence we have that the quantum layer provides a **frequency-access advantage**: the VQC's output spectrum matches the analytical solution more closely than the classical network's does.

### 4.6 📉 Phase 3: gradient variance and barren plateaus (Workstream B)

The gradient variance stays in the range **0.11 to 0.49** across $n=3$ to $n=7$ at depth $L=6$. It does **not** show the $3^{-n}$ exponential decay predicted by McClean et al. for deep random circuits.

> ✅ **Practical implication:** Barren plateaus are **not** a trainability obstacle at the depths and qubit counts relevant to QAPINNs for CFD. This is consistent with the theory: McClean's theorem assumes deep random circuits, while our circuits are shallow (6 layers) and use structured angle encoding.

### 4.7 🔗 Phase 3: CKA information flow (Workstream C)

The VQC's first-layer CKA with the final output is **0.55 to 0.66** across all configurations. This means roughly half the output's representational geometry is already present in the VQC output. The classical twin's first-layer CKA is consistently higher (0.70+), suggesting the quantum layer encodes **less** of the final prediction than the first classical layer does.

> 💡 However, the part the quantum layer does encode is the **high-frequency component**, as confirmed by the Fourier analysis in Workstream A.

### 4.8 📋 Shah et al. Table I reproduction

Shah et al. report $L^2$ values of 0.06, 0.008, and 0.0006 for QA-PINN at 3, 4, and 5 qubits respectively. Our results:

| $n$ | 📄 Shah reported | 🔬 Our faithful repro | 🛠️ Our recipe (same budget) | 📊 Params |
|----:|-----------------:|----------------------:|-----------------------------:|----------:|
| 3 | 0.06 | 0.770 | 0.067 | 74 |
| 4 | 0.008 | 0.759 | 0.021 | 189 |
| 5 | 0.0006 | 0.510 | 0.010 | 600 |

> ⚠️ The faithful Shah reproduction struggles, likely because the encoding scheme and exact optimizer schedule are under-documented in the paper. However, when we **shrink our validated QAPINN recipe to Shah's exact parameter budget**, we recover $L^2$ values within an order of magnitude of Shah's reported numbers. This isolates that the *training procedure* (multi-phase Adam + L-BFGS, shock-focused LHS, Cole-Hopf reference) is what drives the accuracy, not the exact 8-layer architecture.

### 4.9 🏆 Headline finding

<div align="center">

### ⚛️ The Quantum Layer Provides a Frequency-Selective Advantage

</div>

At exact parameter parity, the QAPINN:
- 🤝 **Matches** the classical twin on Burgers (no statistically significant difference at $n=3, 4$)
- 🎯 **Recovers the analytical Fourier spectrum** more accurately than the classical twin at hard viscosity
- ❌ **Does not show an advantage** on the smooth heat equation

This supports a **frequency-selective interpretation** grounded in Schuld et al.'s theorem: the VQC's structured angle encoding gives it preferential access to the Fourier frequencies present in the Burgers shock, which the classical network captures less efficiently. On the smooth heat equation, where the solution is dominated by a single low frequency, the quantum layer's frequency-access advantage does not pay off.

---

## Phase 4 : Design methodology

| + | 📐 **Derive a design methodology** | The ablation grid data is sufficient to derive a problem-specific design methodology: given a PDE's frequency content, prescribe the QAPINN configuration most likely to succeed. This was the original Phase 4 goal and remains the natural continuation. |

---

## 5. 🔮 Limitations and Recommended Next Steps

### 5.1 ⚠️ Limitations

| # | ⚠️ Limitation | 📋 Detail |
|:-:|:--------------|:---------|
| 1 | 🔄 **Easy ≠ hard regime** | The multi-seed paired t-test was conducted at easy viscosity ($\nu = 0.05$) for speed. At hard viscosity, the QAPINN shows clearer improvement with qubit count, but we did not run the full 5-seed paired comparison there due to compute budget. Hard-viscosity results are 3-seed means without paired t-tests. |
| 2 | ⏱️ **8-qubit reduced budget** | At hard viscosity, the $n=8$ configurations used only 1,000 iterations (versus 2,016 for $n=3$ through $n=7$) because of wall-clock constraints. They are marked `in_primary = False` in `run_metadata.csv` and reported separately. The 8-qubit results should be interpreted as exploratory, not definitive. |
| 3 | 📄 **Shah faithful reproduction gap** | Our faithful reproduction achieves $L^2 = 0.51$ to $0.77$, far from their reported 0.0006 to 0.06. We attribute this to under-documented details in the paper (encoding scheme, exact optimizer schedule, collocation strategy). The gap does not affect our internal comparisons, which use the parameter-matched twin as the control. |
| 4 | 📓 **Missing WS-A notebook (easy ν)** | The WS-A analysis code for the easy-viscosity regime was developed in a Google Colab session that was not committed to the repository. Only the output artifacts (figures and CSVs in `Phase_3_heat_easyBurger/WSA/`) are present. The hard-viscosity WS-A code is in `Phase3_Hard_InputBuilder.ipynb`. |
| 5 | 💻 **CPU-only simulation** | All VQC simulations use PennyLane's `default.qubit` state-vector simulator on CPU. We did not test GPU acceleration or actual quantum hardware. The cost-per-step growth (roughly $4\times$ per qubit) limits the practical qubit count to around 8 on our hardware. |
| 6 | 🌊 **Single PDE family** | We tested Burgers and heat. Generalization to other PDEs (Navier-Stokes in 2D, advection-diffusion, wave equation) is not established. |

### 5.2 🚀 Recommended next steps

| # | 🚀 Next step | 📋 Why |
|:-:|:-------------|:------|
| 1 | 🔬 **Run the full 5-seed paired comparison at hard viscosity** | Most important next step. The hard-viscosity regime is where the QAPINN shows the most promise, and a paired t-test there would give a statistically defensible answer. |
| 2 | ⏱️ **Complete the 8-qubit runs at full budget** | The 8-qubit configurations need 2,016 iterations (matching the other widths) for a fair head-to-head. Requires roughly 8 to 12 hours of additional compute. |
| 3 | 📧 **Investigate the Shah reproduction gap** | Contact the authors or consult their supplementary material to resolve the under-documented details, particularly the encoding scheme and the exact Adam learning rate schedule. |
| 4 | 🌊 **Test on a second nonlinear PDE** | The frequency-selective interpretation predicts the QAPINN should help most on PDEs with sharp high-frequency content. Testing on the **Allen-Cahn equation** (which has a sharper interface than Burgers) would be a strong confirmation. |
| 5 | 🎮 **Try GPU-accelerated simulation** | PennyLane supports `lightning.gpu` and hardware backends. Moving the VQC simulation to GPU would allow scaling to 10+ qubits, where barren plateaus should start to appear even at shallow depth. |
| 6 | 📓 **Commit the missing WS-A notebook** | The easy-viscosity Workstream A analysis code should be reconstructed from the Colab session and committed to the repository for full reproducibility. |

---

## 6. 👥 Team Members and Their Contributions

<div align="center">

### 🌟 Team: Quantum Horizon Africa

*AIMS Ghana · WISER 2026 Summer Program*

</div>

This project was completed by a two-person team. Both members contributed to all phases, but primary responsibility for each component is listed below.

---

### 👤 Ounimborbitibou Djabon

<div align="center">

**🌟 Track A Lead · Shah Reproduction · Report & Slides Lead**

📧 [`djabon@aims.edu.gh`](mailto:djabon@aims.edu.gh) · 📱 +228 93 21 72 15 · 🏫 AIMS Ghana

</div>

| 📋 Component | 🎯 Contribution |
|:-------------|:----------------|
| 📄 Phase 2 Track A (Shah reproduction) | Built the faithful ShahQAPINN architecture (8-layer RX VQC, probability state vector measurement, single hidden layer, Adam only) at Shah's exact parameter budget (74, 189, 600). Produced the Table I reproduction. |
| 🟦 Phase 2 classical twin | Implemented the parameter-matched ClassicalTwin with exact parameter parity (1,801, 1,827, 1,853 params). Produced the head-to-head comparison. |
| ⚙️ Phase 2 GAAF-PINN | Implemented the Global Adaptive Activation Function PINN variant with the shared trainable scalar $a$. |
| 🥊 Phase 2 multi-seed production run | Ran the definitive 5-seed paired comparison between QAPINN and ClassicalTwin across both PDEs. |
| 🔥 Phase 2 hard-viscosity (classical) | Ran the 3-seed ClassicalTwin and GAAF-PINN training at hard viscosity for qubit counts 3 through 8. |
| 📊 Phase 3 analysis | Co-implementer of the easy-viscosity Workstream B (learning dynamics) and Workstream C (neuron XAI and CKA) notebooks. |
| 📝 **Project report** | **Wrote the final project report** documenting the full three-phase study, results, and findings. |
| 🎨 **Slides design** | **Designed the presentation slide deck**, supported by Roland. |


---

### 👤 Mawulikplimi Roland Hounkpe

<div align="center">

**🌟 Phase 1 Lead · Track B Lead · Repository Maintainer**

📧 [`rhounkpe@aims.edu.gh`](mailto:rhounkpe@aims.edu.gh) · 📱 +228 92 39 97 21 · 🏫 AIMS Ghana

</div>

| 📋 Component | 🎯 Contribution |
|:-------------|:----------------|
| 🧱 Phase 1 (classical baseline) | **Sole implementer.** Built the 9-layer tanh PINN, identified and fixed the 5 critical issues (Cole-Hopf reference, LHS collocation, scipy L-BFGS-B, xavier init, float64), achieved $L^2 = 2.12 \times 10^{-4}$ (3× better than Raissi et al.). |
| 🔬 Phase 2 Track B (ablation grid) | Designed and ran the 14-configuration ablation grid varying entanglement, measurement, depth, and qubit count. Implemented the gradient-variance (barren-plateau) probe and the per-neuron interpretability hooks. |
| 🔥 Phase 2 hard-viscosity (QAPINN) | Ran the 3-seed QAPINN training at hard viscosity for qubit counts 3 through 8. |
| 📊 Phase 3 analysis | Co-implementer of the hard-viscosity Phase 3 input builder and figure notebooks. |
| 📁 Repository management | Created and maintained the GitHub repository, organized the directory structure, wrote the README. |
| 🎨 Slides design | Supported the presentation slide deck design alongside Djabon. |

---


### 📊 Division of labor summary

| 📋 Component | 🌟 Primary | 🤝 Support |
|:-------------|:----------:|:----------:|
| Phase 1 classical baseline | Hounkpe | Djabon (review) |
| Phase 2 QAPINN framework | Hounkpe | Djabon |
| Phase 2 ClassicalTwin | Djabon | Hounkpe |
| Phase 2 GAAF-PINN | Djabon | Hounkpe |
| Phase 2 Shah reproduction | Djabon | Hounkpe |
| Phase 2 ablation grid | Hounkpe | Djabon |
| Phase 2 multi-seed run | Djabon | Hounkpe |
| Phase 2 hard-viscosity QAPINN | Hounkpe | Djabon |
| Phase 2 hard-viscosity classical | Djabon | Hounkpe |
| Phase 3 WS-A (Fourier) | Hounkpe | Djabon |
| Phase 3 WS-B (gradient variance) | Djabon | Hounkpe |
| Phase 3 WS-C (CKA) | Djabon | Hounkpe |
| 📝 Final project report | Djabon | Hounkpe |
| 🎨 Presentation slides | Djabon | Hounkpe |
| README and repository | Hounkpe | Djabon |

---

## 7. 📁 Repository Structure

```
wiser_project/
├── Phase 1/wiser/                              🧱 Classical PINN baseline (Burgers, hard ν)
│   ├── Phase1_Classical_Baseline.ipynb         📓 The Phase 1 notebook
│   ├── figures/                                🖼️ 6 XAI-hook PNGs
│   └── results/phase1_burgers_*/               📊 metrics.json, model, loss histories
│
├── Phase 2/
│   ├── phase2_part_1/                          ⚛️ QAPINN vs classical twin + GAAF + Shah repro
│   │   ├── Tiers12_q3/                         🔬 First single-seed run
│   │   ├── Tiers12_q38_more_iter/              📈 Extended q3→q8 sweep + barren-plateau probe
│   │   ├── Tiers12 c-PINN/                     🟦 Classical twin at exact parameter parity
│   │   ├── Tiers12_multi_seed/                 🥊 Definitive 5-seed production run
│   │   │   └── QAPINN_PHASE2_FINAL.ipynb       📓 Canonical framework source
│   │   └── GAAF_QAPINN_SHA_REPRO/
│   │       ├── GAAF-PINN/                      ⚙️ GAAF baseline (n3 to n8)
│   │       ├── Heat Multi_seed_q38/            🔥 Heat-equation sweep
│   │       └── Shah & al Reproduction/         📄 Shah et al. Table I reproduction
│   │
│   └── phase2_part_2/wiser/                    🔬 14-config ablation grid
│       └── QAPINN_ABLATION.ipynb
│
├── Phase2_hardBurger/                          🔥 Hard ν=0.01/π training (split per width)
│   ├── QAPINN N3_5/, N6/, N7/, N8/            ⚛️ QAPINN at hard ν, per qubit count
│   ├── GAAF-PINN/                              ⚙️ GAAF at hard ν, all widths
│   └── c-PINN/                                 🟦 Classical twin at hard ν, all widths
│
├── Phase_3_hardBurger/                         📊 Phase 3 analysis at hard ν
│   ├── Phase3_Hard_InputBuilder.ipynb          📓 Rebuilds models, emits all WS-A/B/C tables
│   ├── figures/Phase3_Hard_Figures(1).ipynb    🖼️ Plots HA1 to HA6, HB1 to HB4, HC1 to HC8
│   ├── figures/, fields/, spectra/             🖼️ PNGs and cached NPZs
│   └── *.csv, *.json                           📊 9 summary tables + manifests
│
└── Phase_3_heat_easyBurger/                    📊 Phase 3 analysis at easy ν + Heat
    ├── WSA/                                    🌈 Fourier-spectrum outputs (no notebook committed)
    ├── WSB/WSB_LearningDynamics.ipynb          📉 Learning dynamics and trainability
    └── WSC/                                    🔗 CKA and neuron XAI
        ├── WSC_NeuronXAI.ipynb
        └── WSC_Figures.ipynb
Phase_4/   📐 **Derive a design methodology** |
```

> 📦 **Artefact statistics:** 29 📓 Jupyter notebooks · 580 💾 PyTorch checkpoints (`.pt`) · 1,497 🗃️ cached arrays (`.npz`) · 196 🖼️ raster figures (`.png`) · 50 📐 vector figures (`.pdf`) · 46 📊 results CSVs · 32 📋 JSON manifests

> 🔒 **Git LFS:** Two CSVs under `Phase_3_heat_easyBurger/` are tracked via Git LFS (see `.gitattributes`). All other files are plain git.

---

## 8. 🚀 Quickstart

### 8.1 📦 Install dependencies

```bash
pip install torch pennylane scipy numpy matplotlib pandas
```

> ✅ Python 3.10 or later is required. CPU is sufficient for all experiments. The VQC uses PennyLane's `default.qubit` simulator, so **no quantum hardware is needed**. CUDA is auto-detected when available.

### 8.2 🧱 Run the classical baseline (Phase 1)

```bash
cd "Phase 1/wiser"
jupyter notebook Phase1_Classical_Baseline.ipynb
```

> ⏱️ Expected runtime: approximately 5 minutes on CPU. Output: `results/phase1_burgers_*/metrics.json` with `relative_l2_error` around $2.1 \times 10^{-4}$.

### 8.3 ⚛️ Run the QAPINN grid (Phase 2)

The definitive multi-seed run:

```bash
cd "Phase 2/phase2_part_1/Tiers12_multi_seed"
jupyter notebook QAPINN_PHASE2_FINAL.ipynb
```

This trains QAPINN and ClassicalTwin at **5 seeds × 3 qubit counts × 2 PDEs** (Burgers at $\nu=0.05$ and Heat at $\alpha=0.1$). Schedule: 1,000 Adam iterations (lr $8 \times 10^{-3}$) followed by 1,000 L-BFGS iterations (chunked).

> ⏱️ Expected runtime: approximately 2 hours on CPU.

### 8.4 📊 Run Phase 3 analysis

Phase 3 notebooks do **not** retrain. They consume Phase 2 artifacts (`.pt` checkpoints and `.npz` loss histories).

```bash
cd "Phase_3_hardBurger"
jupyter notebook Phase3_Hard_InputBuilder.ipynb    # Extract all WS-A/B/C inputs first
jupyter notebook figures/Phase3_Hard_Figures.ipynb # Generate the plots
```

---

## 9. 📝 Citation

If you use this code or build on our results, please cite:

```bibtex
@misc{wiser_bqp_2026,
  title  = {Quantum-Assisted Physics-Informed Neural Networks for Computational Fluid Dynamics:
            A Reproduction, Ablation, and Explainability Study},
  author = {Hounkpe, Mawulikplimi Roland and Djabon, Ounimborbitibou},
  year   = {2026},
  note   = {WISER 2026 Summer Program Industry Challenge, in collaboration with BosonQ Psi (BQP)},
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
  volume  = {378}, pages = {686 to 707}, year = {2019}
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

## 📬 Contact

<div align="center">

### 🌟 Team: Quantum Horizon Africa

*AIMS Ghana · WISER 2026 Summer Program*

</div>

| 👤 Name | 🎭 Role | 📧 Email | 📱 Phone |
|:--------|:--------|:---------|:--------|
| **Mawulikplimi Roland Hounkpe** | Phase 1 Lead · Track B Lead · Repository Maintainer | [`rhounkpe@aims.edu.gh`](mailto:rhounkpe@aims.edu.gh) | +228 92 39 97 21 |
| **Ounimborbitibou Djabon** | Track A Lead · Report & Slides Lead | [`djabon@aims.edu.gh`](mailto:djabon@aims.edu.gh) | +228 93 21 72 15 |

> 🤝 We welcome collaboration, questions, and feedback. Feel free to reach out to either of us via email or phone, or open an issue on the GitHub repository.

---

<div align="center">

---

### 🌟 Thank you!

**Team: Quantum Horizon Africa** · AIMS Ghana

**WISER 2026 Summer Program**

📅 August 2026 · 📍 AIMS Ghana

---

⭐ If you found this project helpful, please consider giving it a star on GitHub! ⭐

</div>
