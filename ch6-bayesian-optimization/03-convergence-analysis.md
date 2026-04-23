# 03. BO의 수렴 분석

## 🎯 핵심 질문

- **GP-UCB regret** $R_T \leq \tilde O(\sqrt{T\gamma_T})$ (Srinivas et al. 2010)는 어떻게 유도되는가?
- **Maximum information gain** $\gamma_T$가 왜 kernel에 따라 달라지는가? SE, Matérn에서 각각의 차수?
- Bayesian regret vs frequentist regret?
- BO의 **sample complexity** $O(\epsilon^{-2}$ or $\epsilon^{-d})$ 분석?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

Regret bound는 "**BO가 몇 번 만에 최적에 가까워지는가**"의 수학적 guarantee. 이론 뒷받침 없이는 practitioners가 black-box에 의존. GP-UCB의 sublinear regret은 BO를 **principled exploration method**로 확립한 foundational 결과.

---

## 📐 수학적 선행 조건

- [Ch6-01, 02](./01-gp-bo-framework.md)
- Information theory: mutual information
- Concentration inequalities (Hoeffding, Azuma)
- RKHS (Reproducing Kernel Hilbert Space) 기초

---

## 📖 직관적 이해

### Regret의 의미

$R_T = \sum_{t=1}^T (f(x_t) - f(x^*))$: **누적 regret**. Sub-linear growth ($R_T/T \to 0$)가 "어려운" 목표.

$\tilde O(\sqrt{T\gamma_T})$: $T$가 커져도 **regret당 $T$-factor는 $1/\sqrt T$ 감소**. $\gamma_T$는 "탐색 가능한 정보량".

### Maximum Information Gain $\gamma_T$

$$\gamma_T = \max_{|A|=T} I(f_A; y_A) = \frac{1}{2}\log|I + \sigma_n^{-2}K_A|$$

"어떻게 배치해도 얻을 수 있는 최대 정보량". Kernel의 eigenvalue decay에 의존.

### Kernel별 $\gamma_T$

| Kernel | $\gamma_T$ |
|--------|-----------|
| Linear | $O(d\log T)$ |
| SE (RBF) | $O((\log T)^{d+1})$ |
| Matérn-$\nu$ | $O(T^{d(d+1)/(2\nu + d(d+1))}(\log T)^{...})$ |

**실전 의미**: Smoother kernel → smaller $\gamma_T$ → faster convergence. $d$ 증가 시 exponential growth — curse of dim.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Cumulative Regret

$$R_T = \sum_{t=1}^T r_t, \quad r_t = f(x_t) - f(x^*)$$

(maximization). **Simple regret** $S_T = \min_{t \leq T} r_t$도 자주 사용.

### 정의 3.2 — Maximum Information Gain

$$\gamma_T = \max_{A \subset \mathcal{X}, |A| = T} I(y_A; f_A)$$

Noise $\sigma_n^2$ 가정 하:
$$I(y_A; f_A) = \frac{1}{2}\log|I + \sigma_n^{-2}K(A, A)|$$

### 정의 3.3 — Bayesian vs Frequentist Regret

- **Bayesian**: $\mathbb{E}_{f \sim \text{prior}}[R_T]$
- **Frequentist** (worst-case): $\sup_{f \in \mathcal{H}_k} R_T$ ($\mathcal{H}_k$ = RKHS norm ball)

두 regret 모두 $\tilde O(\sqrt{T\gamma_T})$ 유도 가능 (다른 technical tools).

---

## 🔬 정리와 증명

### 정리 3.1 — GP-UCB Regret Bound (Srinivas et al. 2010)

**명제**: $f \sim \mathcal{GP}(0, k)$, noise $\sigma_n$, UCB with $\kappa_t = \sqrt{2\log(|\mathcal{X}|t^2\pi^2/(6\delta))}$:

**Finite $\mathcal{X}$**:
$$P(R_T \leq \sqrt{C_1 T\kappa_T\gamma_T}) \geq 1 - \delta$$

**Continuous $\mathcal{X} \subset [0, r]^d$** (additional smoothness):

$$P(R_T \leq C_1\sqrt{T\gamma_T\log T}) \geq 1 - \delta$$

**증명 스케치**:

Step 1: UCB $\mu_{t-1}(x) + \kappa\sigma_{t-1}(x)$는 $f(x)$의 high-probability upper bound:
$$P(f(x) \leq \mu_{t-1}(x) + \kappa_t \sigma_{t-1}(x), \forall x) \geq 1 - \delta$$

Step 2: Selected $x_t = \arg\max [\mu + \kappa\sigma]$ implies:
$$\mu_{t-1}(x_t) + \kappa\sigma_{t-1}(x_t) \geq \mu_{t-1}(x^*) + \kappa\sigma_{t-1}(x^*)$$

Step 3: Bound instantaneous regret:
$$r_t = f(x^*) - f(x_t) \leq 2\kappa_t \sigma_{t-1}(x_t)$$

(w.h.p.)

Step 4: Sum of $\sigma$ bounded by information gain:
$$\sum_{t=1}^T \sigma_{t-1}^2(x_t) \leq \frac{2}{\log(1 + \sigma_n^{-2})}\gamma_T$$

Step 5: Cauchy-Schwarz:
$$R_T^2 \leq T \sum r_t^2 \leq T \cdot 4\kappa_T^2 \cdot O(\gamma_T) = O(T\kappa_T\gamma_T)$$

$\square$

### 정리 3.2 — $\gamma_T$ of SE Kernel

**명제**: $k(x, x') = \exp(-\|x-x'\|^2/2\ell^2)$ on $[0, r]^d$:

$$\gamma_T = O((\log T)^{d+1})$$

**증명 sketch**: SE kernel의 eigenvalue가 exponential decay. Mercer expansion → $\sum_i \log(1 + \lambda_i T) \approx$ small. $\square$

**귀결**: SE에선 $R_T = \tilde O(\sqrt{T (\log T)^{d+1}})$ — **poly-logarithmic in $T$**, **폴리 in $d$** (but fixed $d$).

### 정리 3.3 — $\gamma_T$ of Matérn Kernel

**명제**: Matérn-$\nu$ kernel:

$$\gamma_T = O(T^{d(d+1)/(2\nu + d(d+1))}(\log T)^?)$$

$\nu \to \infty$에서 SE로 수렴, $\nu$ 작을수록 rougher function. Smaller $\nu$ → **worse $\gamma_T$** (rough function 탐색 어려움).

### 정리 3.4 — No-Regret Property

**명제**: $\gamma_T = o(T)$이면 $R_T/T \to 0$ (평균 regret 감소):

$$R_T = \tilde O(\sqrt{T\gamma_T}) = o(T) \iff \gamma_T = o(T)$$

- SE: $\gamma_T = (\log T)^{d+1} = o(T)$ ✓
- Linear: $\gamma_T = d\log T = o(T)$ ✓
- Matérn with $\nu$ 매우 작음: 문제 생길 수도

### 정리 3.5 — Bayesian Regret of EI

**명제** (Bull 2011): EI with GP prior:

$$\mathbb{E}[R_T] \leq C\sqrt{T\gamma_T}$$

같은 차수. EI와 UCB가 **asymptotically equivalent**.

**주의**: EI는 $f_{best}$가 너무 과감적 exploration 줄이면 sub-optimal convergence rate (Tran-The et al. 2022 recent study).

### 정리 3.6 — Lower Bound

**명제** (Scarlett 2018): GP-UCB / EI는 **rate-optimal** (log factor 내에서):

$$\inf_{\text{algorithm}}\sup_{f \in \mathcal{H}_k}\mathbb{E}[R_T] \geq \Omega(\sqrt{T\gamma_T})$$

어떤 알고리즘도 이 rate 개선 불가. **UCB가 minimax optimal**.

---

## 💻 실험 — Regret Curve

```python
import numpy as np
import matplotlib.pyplot as plt
import gpytorch
import torch

torch.manual_seed(0); np.random.seed(0)

# Target function (known for experiment, unknown in reality)
def f(x):
    return -(x * np.sin(x) + 0.1*np.cos(5*x))

x_star = -5.0  # approx global min (numerical search)
# Find actual minimum
x_grid_fine = np.linspace(0, 10, 10000)
x_star = x_grid_fine[np.argmin(f(x_grid_fine))]
f_star = f(x_star)

class GPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

def run_bo(n_iter=30, acq='ucb', kappa=2.0, seed=0):
    rng = np.random.default_rng(seed)
    # Init 3 random points
    X = rng.uniform(0, 10, 3)
    y = f(X)
    regret = []
    for t in range(n_iter):
        # Fit GP
        tx = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(y, dtype=torch.float32)
        lik = gpytorch.likelihoods.GaussianLikelihood()
        model = GPModel(tx, ty, lik)
        model.train(); lik.train()
        opt = torch.optim.Adam(model.parameters(), lr=0.1)
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(lik, model)
        for _ in range(100):
            opt.zero_grad()
            loss = -mll(model(tx), ty); loss.backward(); opt.step()
        # Predict
        model.eval(); lik.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            x_grid = torch.linspace(0, 10, 300)
            post = lik(model(x_grid))
            mu = post.mean.numpy()
            sigma = post.variance.sqrt().numpy()
        # Acquisition
        from scipy import stats as st
        if acq == 'ucb':
            a = -(mu - kappa*sigma)  # LCB for min, flip
        elif acq == 'ei':
            z = (y.min() - mu) / np.maximum(sigma, 1e-9)
            a = np.maximum(sigma, 1e-9) * (z*st.norm.cdf(z) + st.norm.pdf(z))
        # Next point
        x_next = x_grid.numpy()[a.argmax()]
        y_next = f(x_next)
        X = np.append(X, x_next); y = np.append(y, y_next)
        # Simple regret
        regret.append(y.min() - f_star)
    return regret

# Compare EI vs UCB vs Random
regrets = {'UCB (κ=2)': [], 'EI': [], 'Random': []}
for seed in range(10):
    regrets['UCB (κ=2)'].append(run_bo(30, 'ucb', 2.0, seed))
    regrets['EI'].append(run_bo(30, 'ei', seed=seed))
    # Random
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, 33); y = f(X)
    reg = [y[:k].min() - f_star for k in range(3, 33)]
    regrets['Random'].append(reg)

fig, ax = plt.subplots(figsize=(10, 5))
for name, rs in regrets.items():
    rs = np.array(rs)
    mean = rs.mean(axis=0); se = rs.std(axis=0)/np.sqrt(len(rs))
    ax.plot(range(len(mean)), mean, label=name, lw=2)
    ax.fill_between(range(len(mean)), mean-se, mean+se, alpha=0.2)
ax.set_yscale('log')
ax.set_xlabel('Iteration'); ax.set_ylabel('Simple regret (log)')
ax.legend(); ax.grid(alpha=0.3)
ax.set_title('BO: EI/UCB vs Random search — sublinear regret')
plt.tight_layout(); plt.savefig('bo_regret.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Sample Complexity of HPO
BO regret이 "**how many GPU hours to find good LR/arch**"의 이론적 뒷받침.

### Safe BO
Constraint-aware BO (예: 분자 설계에서 safety constraint)의 regret bound (Sui 2015).

### Bandit Theory
GP-UCB = infinite-arm contextual bandit. Regret 기술이 discrete bandit에서 차용.

### Multi-objective BO
Pareto frontier hypervolume의 regret bound.

### High-Dim BO (Ch6-04)
$\gamma_T$가 $d$에 지수적 → **intrinsic dimension** 개념 (REMBO, TuRBO 등).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $f$가 GP에서 sample | Worst-case (frequentist)에선 RKHS 가정 |
| Bounded function | Unbounded 시 bound 수정 |
| Known kernel | Unknown kernel → worse rate |
| $\kappa_t$ 이론 schedule | 실전엔 hand-tuned $\kappa$ |
| Low dim $d$ | High-dim exponential cost |

**실전 해석**:
- $d \leq 10$: BO strongly converges, log-poly regret
- $d \sim 20$: 여전히 OK with Matérn-5/2
- $d \gg 20$: special techniques (Ch6-04)

---

## 📌 핵심 정리

$$\boxed{R_T \leq \tilde O(\sqrt{T\gamma_T})}$$

| Kernel | $\gamma_T$ | Regret |
|--------|-----------|--------|
| Linear | $O(d\log T)$ | $\tilde O(\sqrt{dT\log T})$ |
| SE | $O((\log T)^{d+1})$ | $\tilde O(\sqrt{T}(\log T)^{(d+1)/2})$ |
| Matérn-$\nu$ | $O(T^{d(d+1)/(2\nu+d(d+1))})$ | worse for small $\nu$ |

핵심:
- **Sublinear regret** → **no-regret learning**
- $\gamma_T$ = kernel의 information capacity
- UCB가 **minimax optimal**
- Dim $d$에 exponential cost (curse)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Regret $R_T = O(\sqrt T)$이면 "**평균 regret**"은 어떻게 변하나?

<details>
<summary>해설</summary>

$R_T/T = O(1/\sqrt T) \to 0$.

즉 **평균으로 보면 optimal에 접근**. 이것이 "**no-regret**" property — 장기적으론 점점 잘함.

Simple regret $S_T = \min r_t$도 유사한 rate: $O(1/\sqrt T)$.

**Practical**: $T = 100$이면 평균 regret $\sim 0.1 \cdot f$-scale. $T = 10000$이면 $\sim 0.01$.

</details>

**문제 2** (심화): Curse of dimensionality in $\gamma_T$: $d = 30$ SE kernel이면 실질 regret?

<details>
<summary>해설</summary>

$\gamma_T = O((\log T)^{31})$. $T = 1000$: $(\log 1000)^{31} = 7^{31} \approx 10^{26}$ — **이론적으로 terrible**.

실전엔 $\log^{d+1}$ factor가 constant에 흡수되는 경향, 실제 수렴은 poly-logarithmic하지 않음.

**고차원에서 BO가 어려운 이유**:
- $\gamma_T$ 폭발
- Acquisition 최적화 자체가 $d$-dim 문제
- GP 학습도 $d$-dim density 추정

**해결** (Ch6-04):
- Additive kernel: $f = \sum_i f_i$ 가정 → $d \to d'$(small)
- Subspace BO: learn low-dim embedding
- REMBO, TuRBO 등

**Practical $d$** cutoff: SE kernel로 $d = 15$ 이상에서 vanilla BO 실패 시작.

</details>

**문제 3** (AI 연결): LLM hyperparameter tuning (LR, batch size, architecture)에 BO 적용 시 regret bound가 얼마나 의미 있나?

<details>
<summary>해설</summary>

**이론 vs 실전**:
- Regret bound: asymptotic ($T \to \infty$). 실전 HPO는 $T = 50$-$100$ evaluations.
- Bound의 constant factor가 커서 finite-$T$ guidance가 제한적.

**그래도 의미**:
- **Sublinear** convergence 자체가 "BO worth doing" 신호
- Kernel choice (Matérn-5/2 > SE for typical ML)의 이론 근거
- Batch size·parallelization의 effective $T$ 스케일링

**실전 best practice**:
- $d \leq 20$ HPO: BO (Ax, BoTorch, Optuna)
- $d > 20$: hybrid (BO for key params, random for rest)
- Heuristic: **Bayesian priors on good defaults** (e.g., log-uniform LR)

LLM community에선 **Population Based Training**이나 **HyperBand**가 BO와 경쟁 — 각자 다른 regret characterization. 
Theory helps choose, but empirical evaluation의 final judge.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Acquisition Function들](./02-acquisition-functions.md) | [📚 README로 돌아가기](../README.md) | [04. BO의 실전 확장 ▶](./04-bo-extensions.md) |

</div>
