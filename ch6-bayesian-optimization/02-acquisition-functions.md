# 02. Acquisition Function들

## 🎯 핵심 질문

- **Expected Improvement (EI)** $\text{EI}(x) = \sigma(x)(z\Phi(z) + \phi(z))$는 어떻게 유도되는가?
- **UCB** $\mu(x) + \kappa\sigma(x)$, **Thompson Sampling (TS)**, **PI**의 수학적 구조는?
- 각각의 **exploration-exploitation trade-off**를 어떻게 조절?
- 어느 상황에서 어느 acquisition이 적합한가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

Acquisition 선택이 BO 성능의 **가장 큰 결정 요인**. EI는 default choice, UCB는 **regret bound** 증명 용이, TS는 **batch parallelization** 자연, PI는 historical. BoTorch·Ax의 핵심 library가 이 함수들의 구현. Bayesian optimization의 "탐색 철학"을 수학적으로 구현.

---

## 📐 수학적 선행 조건

- [Ch6-01 GP BO](./01-gp-bo-framework.md)
- [Ch1-04 Predictive](../ch1-bayesian-foundation/04-predictive-distribution.md)
- Standard normal pdf $\phi$, cdf $\Phi$
- Truncated Gaussian expectation

---

## 📖 직관적 이해

### 네 가지 주요 Acquisition

| | 수식 | Exploration | Exploitation |
|---|---|---|---|
| **EI** | $\mathbb{E}[\max(0, f_{best} - f(x))]$ | Moderate | Strong |
| **UCB** | $\mu(x) + \kappa\sigma(x)$ | Adjustable ($\kappa$) | With mean |
| **TS** | Sample $\tilde f$, argmax | Stochastic | Posterior-weighted |
| **PI** | $P(f(x) < f_{best})$ | Weak | Over-greedy |

### 각각의 철학

- **EI**: "**평균적으로 얼마나 개선**될까" — balanced
- **UCB**: "**낙관적 upper bound** 아래서 움직임" — optimism in uncertainty
- **TS**: "**random posterior realization**의 optimum" — stochastic
- **PI**: "**개선 확률**만" — greedy, exploration 부족

### 요리 비유

- EI: "기대 맛 증가분"
- UCB: "최대 잠재력"
- TS: "한 가지 가능한 시나리오의 최고 메뉴"
- PI: "내 기록 깰 확률"

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Expected Improvement (EI)

**Minimization** (minimum 찾기):
$$\text{EI}(x) = \mathbb{E}[\max(0, f_{best} - f(x))]$$

$f_{best}$ = best observed so far.

### 정의 2.2 — Upper Confidence Bound (UCB)

**Minimization**:
$$\text{UCB}(x) = \mu(x) - \kappa\sigma(x)$$

(**LCB** for min). $\kappa \geq 0$: exploration weight. **Next** = argmin.

### 정의 2.3 — Thompson Sampling (TS)

1. Sample $\tilde f \sim \mathcal{GP}(\mu_n, k_n)$ (posterior function sample)
2. $x_{next} = \arg\min \tilde f$

Randomized — different sample each iter.

### 정의 2.4 — Probability of Improvement (PI)

$$\text{PI}(x) = P(f(x) < f_{best} - \xi)$$

$\xi \geq 0$ trade-off parameter.

---

## 🔬 정리와 증명

### 정리 2.1 — EI의 Closed Form

**명제**: Posterior $f(x) \sim \mathcal{N}(\mu(x), \sigma^2(x))$이면:

$$\text{EI}(x) = \sigma(x)[z\Phi(z) + \phi(z)]$$

where $z = (f_{best} - \mu(x))/\sigma(x)$.

**증명**:

$$\text{EI}(x) = \mathbb{E}[\max(0, f_{best} - f(x))]$$

$Y := f_{best} - f(x) \sim \mathcal{N}(f_{best} - \mu(x), \sigma^2(x))$.

$\max(0, Y)$ = truncated $Y$:
$$\mathbb{E}[\max(0, Y)] = (f_{best} - \mu)\Phi\left(\frac{f_{best} - \mu}{\sigma}\right) + \sigma\phi\left(\frac{f_{best} - \mu}{\sigma}\right)$$

$z = (f_{best} - \mu)/\sigma$:
$$= \sigma z \Phi(z) + \sigma\phi(z) = \sigma[z\Phi(z) + \phi(z)]$$

$\square$

**해석**:
- $\sigma(x)$ 높음 (unexplored) → EI 증가 (exploration)
- $\mu(x)$ 낮음 (promising) → $z$ 큼 → $\Phi(z) \to 1$ → EI 증가 (exploitation)

### 정리 2.2 — UCB의 Regret Bound 기초

**명제** (Srinivas et al. 2010): 적절한 $\kappa_t$ 선택으로 GP-UCB의 cumulative regret:

$$R_T = \sum_t (f(x_t) - f(x^*)) \leq \tilde O(\sqrt{T\gamma_T})$$

$\gamma_T$ = maximum information gain (kernel-dependent).

**증명**: Ch6-03에서 자세히. Core idea: $\mu + \kappa\sigma$가 true $f$의 high-probability upper bound, regret은 $\sum \sigma(x_t) \leq O(\sqrt{T\gamma_T})$. $\square$

### 정리 2.3 — Thompson Sampling Regret

**명제** (Russo & Van Roy 2014): Bayesian cumulative regret of TS:

$$\mathbb{E}[R_T] \leq \sqrt{T \cdot (|\mathcal{X}| \cdot \log T)}$$

for finite $\mathcal{X}$. GP case: $\sqrt{T\gamma_T}$-like.

**증명**: Information-theoretic analysis. $\square$

**실전 장점**: 자연스러운 **batch parallelization** — multiple TS samples를 parallel evaluate.

### 정리 2.4 — PI의 한계

**명제**: PI $= \Phi((f_{best} - \mu)/\sigma)$는 **exploration을 놓치기 쉬움**.

**이유**: 이미 개선 확률 높은 영역에 집중. $\sigma$ 고려 weight 작음.

**수정**: $\text{PI}(x; \xi) = P(f < f_{best} - \xi)$, $\xi > 0$으로 소폭 improvement만 보상.

### 정리 2.5 — 각 Acquisition의 Limiting Behavior

| | $\sigma(x) \to 0$ | $\sigma(x) \to \infty$ |
|---|---|---|
| EI | $\max(0, f_{best} - \mu)$ | $\sigma\phi(0) = \sigma/\sqrt{2\pi}$ (linear in $\sigma$) |
| UCB | $\mu$ | $-\infty$ (강한 exploration) |
| PI | $0/1$ step function | $1/2$ |

EI는 **adaptive** (두 축을 균형).

---

## 💻 구현

```python
import numpy as np
from scipy import stats

def ei(mu, sigma, f_best, xi=0.01):
    """Expected Improvement (minimization)."""
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu - xi) / sigma
    return sigma * (z * stats.norm.cdf(z) + stats.norm.pdf(z))

def ucb(mu, sigma, kappa=2.0):
    """Lower CB (minimization)."""
    return mu - kappa * sigma

def pi(mu, sigma, f_best, xi=0.01):
    """Probability of Improvement."""
    sigma = np.maximum(sigma, 1e-9)
    z = (f_best - mu - xi) / sigma
    return stats.norm.cdf(z)

def thompson_sample(gp_mean_func, gp_samp_func, x_grid, rng):
    """Sample one posterior realization from GP over grid, return argmin."""
    f_sample = gp_samp_func(x_grid)  # GP draw
    return x_grid[f_sample.argmin()]

# ────────────────────────────────────────────────
# BO loop (pseudo-code)
# ────────────────────────────────────────────────
# X_obs, y_obs = initial_design()
# for t in range(T):
#     gp.fit(X_obs, y_obs)
#     mu, sigma = gp.predict(x_grid)
#     a = ei(mu, sigma, y_obs.min())
#     x_next = x_grid[a.argmax()]
#     y_next = f(x_next)
#     X_obs.append(x_next); y_obs.append(y_next)
```

---

## 🔗 AI/ML 연결

### BoTorch
Monte Carlo EI (qEI), UCB, TS가 built-in. Batch versions for parallel.

### Information-theoretic Acquisitions
**Max-value Entropy Search (MES)**, **Predictive Entropy Search (PES)** — 정보이론적 확장.

### Contextual Bandit
UCB와 TS가 bandit literature에서 온 것. BO는 "**continuous-arm contextual bandit**"의 GP version.

### Multi-objective BO
**qEHVI** (Hypervolume Expected Improvement) — Pareto frontier에 대한 acquisition.

### Constraint BO
**EIC** (EI with Constraints) — 제약 조건 있는 최적화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| GP posterior 정확 | Bad kernel → bad acquisition |
| $\mu, \sigma$ tractable | Non-Gaussian observation 시 복잡 |
| Single objective | Multi-obj에는 수정 필요 |
| $f$ deterministic | Noise → EI에 $\sigma_n$ 추가 |
| $\xi$ hand-tune (for PI) | 데이터별 sensitivity 있음 |

**실무 가이드**:
- **Default**: EI (안정적, 잘 작동)
- **Regret 증명 필요**: UCB
- **Batch parallel**: TS
- **Avoid**: PI (over-greedy)

---

## 📌 핵심 정리

$$\boxed{\text{EI}(x) = \sigma(x)[z\Phi(z) + \phi(z)], \quad z = (f_{best} - \mu(x))/\sigma(x)}$$

$$\boxed{\text{UCB}(x) = \mu(x) - \kappa\sigma(x)}$$

$$\boxed{\text{TS: } x = \arg\min \tilde f, \tilde f \sim \mathcal{GP}(\mu, k)}$$

선택 가이드:
- **EI**: 안전한 default
- **UCB**: regret 증명 가능
- **TS**: batch parallel
- **PI**: avoid (exploration 약)

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\sigma(x) = 0$ (fully observed)이면 EI 값은?

<details>
<summary>해설</summary>

$\sigma \to 0$, $z = (f_{best} - \mu)/\sigma$:

- $\mu < f_{best}$: $z \to +\infty$, $\Phi(z) \to 1, \phi(z) \to 0$. $\text{EI} = \sigma z \cdot 1 \to (f_{best} - \mu) > 0$.
- $\mu > f_{best}$: $z \to -\infty$, $\Phi \to 0, \phi \to 0$. $\text{EI} = 0$.
- $\mu = f_{best}$: $z = 0/0$ limit. Careful L'Hopital: $\text{EI} = \sigma\phi(0) = \sigma/\sqrt{2\pi} \to 0$.

즉 fully observed + $\mu < f_{best}$ → EI = improvement amount, exploit하라는 신호.
Fully observed + $\mu \geq f_{best}$ → EI = 0, 가지 말라는 신호.

자연스러운 "**해당 점에 예상 이익**" 해석.

</details>

**문제 2** (심화): UCB의 $\kappa$를 **시간 $t$에 따라 증가**시키는 이유?

<details>
<summary>해설</summary>

Srinivas et al. 2010의 theoretical schedule:

$$\kappa_t = \sqrt{2\log(|\mathcal{X}|\pi^2 t^2/6\delta)}$$

(finite $\mathcal{X}$의 경우).

**이유**:
- 초기엔 data 적음 → $\sigma$ 크므로 UCB에 반영
- 시간 지남 → exploration 유지 위해 **$\kappa$ 조금씩 증가** (로그 scale)
- 이로써 모든 $x$가 infinitely visited (consistency)

**실전**: 
- $\kappa = 2$ 고정 (approximately 95% CI — "Zadeh's rule")
- Schedule은 이론 증명에 중요, empirical 성능은 constant도 OK

$\kappa$ 너무 작 (< 1): exploit heavy, local optimum 위험.
$\kappa$ 너무 큼: 낭비.

</details>

**문제 3** (AI 연결): 딥러닝 HPO에 TS가 **자연스럽게 병렬**인 이유?

<details>
<summary>해설</summary>

**TS**: 매번 new posterior sample → **different $x_{next}$** (stochastic).

Batch parallelization:
```
1. Draw q samples f_1, ..., f_q from posterior
2. x_i = argmin f_i for each i
3. Evaluate f at x_1, ..., x_q in parallel
```

각 $x_i$가 independent posterior sample로부터 → **natural diversity**.

**EI는 batch 복잡**: naive batch EI는 same point 여러 번 선택 위험. **qEI** (BoTorch)는 이를 joint로 최적화 — 계산 heavy.

**HPO in practice**:
- Cloud evaluation (여러 GPU 동시) → batch BO 필수
- Ax(Facebook)/Optuna의 parallel strategy에 TS 기반 흐름 많음
- **Async BO**: evaluation 시간 불균일할 때 특히 TS 효과적

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. GP 기반 BO 프레임워크](./01-gp-bo-framework.md) | [📚 README로 돌아가기](../README.md) | [03. BO의 수렴 분석 ▶](./03-convergence-analysis.md) |

</div>
