# 04. BO의 실전 확장

## 🎯 핵심 질문

- **High-dimensional BO** — REMBO, LineBO, TuRBO 등의 전략?
- **Multi-fidelity BO** — FABOLAS와 cheap-expensive evaluation 결합?
- **Bayesian Quadrature** — BO의 integration version?
- 딥러닝 **hyperparameter tuning** 에서 실전 wall-clock trade-off?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

실제 BO 응용은 종종 표준 가정 밖: $d = 50$ NN hyperparams, epoch 수가 조절 가능 (multi-fidelity), noise 정량 미지, batch parallel evaluation. 각 상황에 맞는 **BO의 변형**들이 실전 성능 결정. BoTorch, Ax의 실전 recipe들의 이론적 기반.

---

## 📐 수학적 선행 조건

- [Ch6-01~03](./01-gp-bo-framework.md)
- Low-rank matrix decomposition
- Trust region methods

---

## 📖 직관적 이해

### High-Dim 문제

$\gamma_T$의 exponential growth with $d$ (Ch6-03) → vanilla BO 실패.

**전략들**:
1. **Random projection**: REMBO — $x = A z, z \in \mathbb{R}^{d_e}$ with small $d_e$
2. **Line search**: LineBO — 1D BO along random directions
3. **Trust region**: TuRBO — 현재 best 주변 small box에 BO
4. **Additive kernel**: $f = \sum f_i(x_i)$ 가정

### Multi-Fidelity

Same $f$에 대한 **cheap proxies**:
- ML: 작은 dataset, 적은 epochs, 작은 model
- Physics: coarser mesh
- Etc.

**FABOLAS** (Klein 2017): dataset size를 fidelity로, BO가 size·params 동시 최적화.

### Bayesian Quadrature

$\int f(x) p(x) dx$ (expectation) 최적화. BO와 유사하지만 **max → integral**.

### 요리 비유

- High-dim: "많은 재료 중 **중요한 몇 개만** 주로 조절"
- Multi-fidelity: "**미니 시식** → 본격 조리"
- BQ: "**평균 맛**에만 관심"

---

## ✏️ 엄밀한 정의

### 정의 4.1 — REMBO (Random EMbedding BO)

$$f: \mathbb{R}^d \to \mathbb{R}, \quad x = A z, \quad z \in \mathbb{R}^{d_e}$$

$A$ random matrix, $d_e \ll d$. BO on $z$-space with effective dim $d_e$.

### 정의 4.2 — TuRBO

- 현재 best $x^*$ 주변 **trust region** $B(x^*, r)$
- TR 내부에서 local GP + TS
- TR을 success/failure 따라 확장/축소
- Multi-trust-region for exploration

### 정의 4.3 — Multi-Fidelity Objective

$$f(x, s) : x \in \mathcal{X}, s \in [0, 1]$$

$s = 1$: full fidelity (expensive), $s < 1$: cheap approximation.

**Goal**: $\max f(x, 1)$ with **budget** $\int c(s) ds$ 제약.

### 정의 4.4 — Bayesian Quadrature

$$I = \int f(x) p(x) dx$$

GP posterior on $f$ → analytic posterior on $I$ (linear functional).

---

## 🔬 정리와 경험적 결과

### 정리 4.1 — REMBO의 Effective Dimension

**명제** (Wang et al. 2016): $f$가 **effective dim $d_e$** 함수 ($d_e$ direction만 영향, 나머지는 redundant)이면:

$$R_T = \tilde O(\sqrt{T\gamma_T(d_e)}) \quad \text{with } d_e \text{ instead of } d$$

Random Gaussian $A$로 $d_e$-dim subspace에 "충분한 확률로" 포함.

**제약**: $d_e$ 사전 지정 필요, 실제 $d_e$ 모르면 over/under-specify 위험.

### 정리 4.2 — TuRBO의 Empirical Performance

**명제** (Eriksson et al. 2019): TuRBO가 표준 BO를 **100D+까지** 확장.

**이유**:
- 전역 BO 대신 **local GP** → 차원 효과 완화
- Multi-trust-regions로 exploration
- Thompson sampling batch evaluation

**실제 NN HPO**: 50~500 params에서 TuRBO가 random search 대비 10× faster.

### 정리 4.3 — Multi-Fidelity Acquisition

**EIC-M (Expected Improvement with Cost)**:
$$\alpha(x, s) = \frac{\text{EI}(x, s)}{c(s)}$$

cost-normalized EI. Cheap evaluation으로 high-information 얻으면 선호.

**FABOLAS**: fidelity-aware GP with $s$ as input, learns $f(x, s)$ joint.

### 정리 4.4 — Bayesian Quadrature Posterior

**명제**: $f \sim \mathcal{GP}(\mu, k)$, weights $w(x) = p(x)$:

$$I|y \sim \mathcal{N}(\mu_I, \sigma_I^2)$$

Closed form from linearity:
$$\mu_I = \int \mu(x)p(x)dx$$
$$\sigma_I^2 = \int\int[k(x, x') - k_n(x, x')]p(x)p(x')dxdx'$$

**장점**: Integration 정확도가 MC ($O(T^{-1/2})$)보다 빠르게 수렴 (kernel dependent, up to $O(T^{-k})$).

---

## 💻 BoTorch 실전

```python
import torch
import botorch
from botorch.models import SingleTaskGP, FixedNoiseGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition import qExpectedImprovement, UpperConfidenceBound
from botorch.optim import optimize_acqf
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np

# Define black-box
def objective(X):
    # X: (n, d) tensor; returns (n, 1) tensor
    return -(X**2).sum(dim=-1, keepdim=True) + 0.1*torch.sin(5*X.sum(dim=-1, keepdim=True))

d = 5  # moderate dim
bounds = torch.stack([-2*torch.ones(d), 2*torch.ones(d)])

# Initial design (Sobol)
from botorch.utils.sampling import draw_sobol_samples
X = draw_sobol_samples(bounds=bounds, n=10, q=1).squeeze(1)
Y = objective(X)

# BO loop with qEI (batch)
for t in range(20):
    gp = SingleTaskGP(X, Y)
    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)
    
    # Batch EI (q=4 parallel)
    qEI = qExpectedImprovement(gp, best_f=Y.max())
    candidates, _ = optimize_acqf(qEI, bounds=bounds, q=4, num_restarts=5, raw_samples=256)
    
    new_Y = objective(candidates)
    X = torch.cat([X, candidates], dim=0)
    Y = torch.cat([Y, new_Y], dim=0)
    print(f"iter {t}: best = {Y.max().item():.4f}")

# 50 → 90 evaluations, near-optimal
```

### Optuna for HPO

```python
import optuna

def train_model(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    hidden = trial.suggest_int('hidden', 32, 512, log=True)
    dropout = trial.suggest_uniform('dropout', 0.0, 0.5)
    # train & return val_loss
    # ...
    return val_loss

study = optuna.create_study(direction='minimize', 
                             sampler=optuna.samplers.TPESampler())  # TPE = tree-structured Parzen, BO-like
study.optimize(train_model, n_trials=100)
print(study.best_params)
```

TPE (Bergstra 2013): BO variant, popular for HPO.

---

## 🔗 AI/ML 연결

### Deep Learning HPO
**Ax** (Meta), **BoTorch**, **Optuna**, **Ray Tune** — 모든 현대 HPO framework가 BO 기반 + multi-fidelity + batch.

### Architecture Search
**NAS with BO**: discrete/graph-structured $\mathcal{X}$에 custom kernel.

### AutoML
**AutoSklearn, AutoGluon** — BO가 model + hyperparam selection 통합.

### Drug Discovery
High-dim (분자 fingerprint) BO with graph neural network kernel.

### Gaussian Process for LLMs
Finetune param tuning with small number of GPUs — BO efficient under budget.

---

## ⚖️ 가정과 한계

| 기술 | 장점 | 단점 |
|------|------|------|
| REMBO | Effective dim 활용 | $d_e$ 사전 지정 |
| LineBO | Simple 1D subproblem | Direction 무작위 |
| TuRBO | 100+D 가능 | Trust region tuning |
| Multi-fidelity | 비용 절감 | Fidelity model 필요 |
| BQ | Integration 빠름 | Max보다 application 적음 |

**Practical rule**:
- $d \leq 20$: vanilla BO (BoTorch)
- $d = 20 \sim 100$: TuRBO, SAASBO (sparse kernel)
- $d > 100$: 도메인 특화 (분자, NAS)

---

## 📌 핵심 정리

| 문제 | 해결 |
|------|------|
| High-dim | REMBO, TuRBO, additive kernel |
| Multi-fidelity | FABOLAS, EIC-M |
| Batch parallel | qEI, TS-batch |
| Constraints | EIC (EI with Constraints) |
| Multi-objective | qEHVI |
| Integration | Bayesian Quadrature |

---

## 🤔 생각해볼 문제

**문제 1** (기초): TuRBO에서 trust region을 언제 확장/축소?

<details>
<summary>해설</summary>

**확장**: $k$ successful trials (improvement 발생) → $r \leftarrow 2r$.

**축소**: $k$ failed trials → $r \leftarrow r/2$.

"Heating-cooling" schedule. TR 크기가 **local landscape에 adaptive**.

$r_{\min}$ 도달하면 TR 재시작 (new random center).

Multi-TR: 여러 TR을 병행 → robust to bad initialization.

</details>

**문제 2** (심화): FABOLAS에서 dataset size $s$를 fidelity로 사용할 때 GP 설계?

<details>
<summary>해설</summary>

Joint kernel:
$$k((x, s), (x', s')) = k_x(x, x') \cdot k_s(s, s')$$

- $k_x$: hyperparam kernel (e.g., RBF)
- $k_s$: fidelity kernel, **decay** in $s'-s$ → 큰 $s$에서 observations가 작은 $s$로도 **partial info**

학습:
- 작은 $s$ (cheap): quick info
- 큰 $s$ (expensive): actual target

Acquisition **cost-aware**: $\alpha = \text{EI}(x, s=1) / \text{cost}(s)$ 등.

**결과**: Empirically 5-10× speedup vs naive BO (same wall-clock).

</details>

**문제 3** (AI 연결): GPT fine-tuning의 **learning rate**를 BO로 어떻게 찾을까?

<details>
<summary>해설</summary>

**Setup**:
- $x = \log(\text{LR}) \in [-10, -2]$
- $f(x) = $ validation loss after fine-tuning
- Each evaluation: 수 시간 GPU

**Multi-fidelity**: 
- $s = $ training steps (short → long)
- Early-stopping + BO → HyperBand+BOHB

**Batch**: 4개 GPU → $q = 4$ parallel EI

**Domain**: LR은 log-uniform, 다른 params (warmup, batch size) joint

**Practical code** (Optuna):
```python
def objective(trial):
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-3)
    warmup = trial.suggest_int('warmup', 0, 1000)
    # fine-tune and return val_loss
    return val_loss

study = optuna.create_study(sampler=TPESampler(), pruner=HyperbandPruner())
study.optimize(objective, n_trials=50, timeout=24*3600)
```

**효과**: Random search의 1/4 budget으로 optimal 근접.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. BO의 수렴 분석](./03-convergence-analysis.md) | [📚 README로 돌아가기](../README.md) | [Ch7-01. Diffusion Model의 Bayesian 해석 ▶](../ch7-advanced-topics/01-diffusion-bayesian.md) |

</div>
