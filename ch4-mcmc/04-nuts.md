# 04. No-U-Turn Sampler (NUTS)

## 🎯 핵심 질문

- HMC의 두 hyperparameter — **step size $\epsilon$**와 **trajectory length $L$** — 를 어떻게 자동 조정하는가?
- **U-turn**이란 무엇이고 왜 trajectory 종료의 신호인가?
- NUTS의 **binary tree doubling** 알고리즘과 detailed balance 유지 방법은?
- 왜 NUTS가 Stan·PyMC의 기본 sampler인가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

수동 hyperparameter 튜닝은 **practical Bayesian inference의 큰 장벽**이었다. NUTS (Hoffman & Gelman 2014)는 이를 완전 자동화 — "모델 작성 → run → posterior sample" pipeline을 가능케 함. Probabilistic programming(Stan, PyMC, NumPyro, Pyro)의 commercial success에 직접 기여. BNN·hierarchical model·state-space model에서의 practical MCMC 표준.

---

## 📐 수학적 선행 조건

- [Ch4-03 HMC](./03-hamiltonian-monte-carlo.md)
- Dual averaging (Nesterov 2009) — step size adaptation
- Reversible Markov chain 기초

---

## 📖 직관적 이해

### HMC의 튜닝 문제

HMC 두 knob:
1. **$\epsilon$** (step size): acceptance와 accuracy trade
2. **$L$** (num leapfrog steps): trajectory length

수동 튜닝:
- 너무 짧은 trajectory → mixing 느림
- 너무 긴 trajectory → U-turn(되돌아옴) 낭비
- 적절한 $\epsilon$: acceptance $\sim 0.65$

### U-turn Detection

Trajectory가 **되돌아오기 시작**하면 중단. 수학적으로:

$$(\theta_+ - \theta_-)^T p_- < 0 \quad \text{또는} \quad (\theta_+ - \theta_-)^T p_+ < 0$$

양쪽 방향 모두 "**반대 방향**"을 가리킴 = U-turn.

### 아이디어 — Doubling

1. Momentum $p$ 샘플
2. Forward/backward **양쪽으로 trajectory** 확장, doubling (1 → 2 → 4 → 8 → ...)
3. U-turn 감지되면 중단
4. 전체 trajectory에서 uniform하게 하나 sample

### 요리 비유

"어느 방향으로 얼마나 밀지 미리 몰라 — 양쪽으로 **점점 크게 확장하면서 탐색**하고, 길이 꺾이는 순간 멈춤. 그중 무작위 하나를 뽑음."

---

## ✏️ 엄밀한 정의

### 정의 4.1 — U-turn Criterion

Trajectory $(\theta_-, p_-) \to \cdots \to (\theta_+, p_+)$의 "U-turn":

$$(\theta_+ - \theta_-)^T p_- < 0 \quad \text{or} \quad (\theta_+ - \theta_-)^T p_+ < 0$$

역방향으로 움직이기 시작 = 탐색 효과 감소.

### 정의 4.2 — NUTS Binary Tree

Tree depth $j$에서 $2^j$ leapfrog steps. Recursive:
- Level 0: 1 step
- Level $j$: 2 sub-trees of depth $j-1$ concatenated (direction random)

각 sub-tree 내부 U-turn 체크 + overall tree U-turn 체크.

### 정의 4.3 — NUTS Proposal

**Slice sampling extension**: auxiliary $u \sim \text{Unif}(0, \exp(-H(\theta_0, p_0)))$ 도입. "valid" state $= \{(\theta, p) : e^{-H} \geq u\}$.

Tree를 U-turn 또는 unstable state까지 확장. Valid states 중 **multinomial sampling**.

### 정의 4.4 — Dual Averaging for $\epsilon$

Warmup 동안 step size를 **target acceptance rate** $\delta$ (보통 $0.8$)에 맞춰 조정:

$$\epsilon_{t+1} \leftarrow f(\epsilon_t, \text{statistics of acceptances})$$

Nesterov's dual averaging:
$$\log\epsilon_t = \mu - \frac{\sqrt t}{\gamma}\bar H_t, \quad \bar H_{t+1} = (1 - 1/t)\bar H_t + (\delta - \alpha_t)/t$$

Robbins-Monro type scheme.

---

## 🔬 정리와 증명

### 정리 4.1 — NUTS의 Detailed Balance

**명제**: NUTS transition은 정상분포 $\pi_{\text{joint}}(\theta, p) \propto e^{-H}$를 유지.

**증명 스케치** (Hoffman & Gelman 2014 Theorem 1):

Slice variable $u$로 augment. Tree 구성이 **reversible** (양방향 doubling). 각 valid leaf node에서 uniform sampling → detailed balance 유지.

증명 핵심:
- Tree 구성의 reversibility
- U-turn criterion의 대칭성
- Slice sampling의 probabilistic validity

$\square$

**귀결**: Acceptance가 **암묵적** — 별도 accept/reject 없이 trajectory 자체가 correct sample 제공.

### 정리 4.2 — U-turn Criterion의 Rotation Invariance

**명제**: U-turn 판정 $(\theta_+ - \theta_-)^T p_\pm < 0$은 momentum 회전에 **불변**.

**증명**: Inner product preserves under orthogonal rotation. $\square$

**귀결**: $M$ 선택(mass matrix)에 대해 robust.

### 정리 4.3 — Dual Averaging 수렴

**명제**: Nesterov (2009) dual averaging 기반 $\epsilon$ adaptation은:

$$\mathbb{E}[\alpha(\epsilon_t)] \to \delta$$

(target acceptance rate로 수렴).

**증명**: Stochastic approximation theory. $\sum 1/t = \infty$ ensures exploration, asymptotic unbiased. $\square$

### 정리 4.4 — NUTS Max Tree Depth Limit

**실전**: Tree depth $j_{\max}$ (보통 10 → $2^{10} = 1024$ leapfrog steps) 제한. 이 limit 도달 = trajectory가 "매우 긴" 또는 **degenerate** posterior 신호.

---

## 💻 PyMC 실전 예제

```python
# NumPy로 NUTS 직접 구현은 복잡 (Hoffman-Gelman Algorithm 3)
# 실전은 PyMC/Stan/NumPyro 사용
import pymc as pm
import numpy as np
import arviz as az

# ────────────────────────────────────────────────
# Bayesian hierarchical linear regression
# ────────────────────────────────────────────────
rng = np.random.default_rng(0)
n_groups, n_per_group = 8, 30
group_idx = np.repeat(np.arange(n_groups), n_per_group)
x = rng.standard_normal(n_groups*n_per_group)
beta_true = rng.normal(1.0, 0.3, n_groups)
y = beta_true[group_idx]*x + 0.5*rng.standard_normal(len(x))

with pm.Model() as model:
    mu_beta = pm.Normal('mu_beta', 0, 5)
    sigma_beta = pm.HalfNormal('sigma_beta', 1)
    beta = pm.Normal('beta', mu_beta, sigma_beta, shape=n_groups)
    sigma = pm.HalfNormal('sigma', 1)
    y_obs = pm.Normal('y', mu=beta[group_idx]*x, sigma=sigma, observed=y)
    
    idata = pm.sample(2000, tune=1000, target_accept=0.9, 
                      return_inferencedata=True, random_seed=0)

# Diagnostics
print(az.summary(idata, var_names=['mu_beta', 'sigma_beta', 'sigma']))
print(f"\nMax R-hat: {az.rhat(idata).max():.3f}")
print(f"Min ESS: {az.ess(idata).min():.0f}")

# Plot trace
import matplotlib.pyplot as plt
az.plot_trace(idata, var_names=['mu_beta', 'sigma_beta'])
plt.tight_layout(); plt.savefig('nuts_pymc.png', dpi=150); plt.show()
```

**관찰**: 단 **3줄로 NUTS + adaptation + diagnostics**. User 관심은 **모델 설계** on, 추론 세부는 PyMC/Stan가 처리.

---

## 🔗 AI/ML 연결

### Stan / PyMC / NumPyro / Pyro
모두 NUTS를 기본 sampler로. Model as Python code → NUTS → posterior.

### BNN (Limited)
소규모 BNN에 NUTS 시도 가능. 수천 파라미터까지. 그 이상은 SVI/Laplace.

### Hierarchical Model
Multi-level regression, random-effect model — NUTS가 강력 (correlation 많음).

### Gaussian Process Regression
Hyperparameter posterior (length-scale, variance) → NUTS.

### Non-centered Parameterization
Funnel-shaped posterior(hierarchical σ) → **$\beta = \mu + \sigma\tilde\beta$** 재매개변수화로 NUTS friendly.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Differentiable posterior | Discrete params(Bernoulli) → MH 자동 fallback |
| Reasonable dimensionality | 초대규모 $10^6$ BNN에선 여전히 느림 |
| Continuous params | Integer/categorical variables 직접 불가 |
| Funnel 등 pathological posterior | Non-centered 재매개변수 필요 |
| Memory for tree | $j_{\max} = 10$이면 trajectory 1024 states — memory heavy |

**실무 팁**:
- `target_accept = 0.9` (default): 보수적, 고품질
- `target_accept = 0.95`: 더 엄격 (divergence 적음)
- **Divergence** 발생 → step size 너무 크거나 funnel → 재매개변수화 검토
- `max_treedepth = 10` 초과하면 posterior 성격 검토

---

## 📌 핵심 정리

$$\boxed{\text{NUTS} = \text{HMC} + \text{자동 } \epsilon + \text{자동 trajectory (U-turn 감지)}}$$

핵심:
- **Doubling tree** + U-turn stop
- **Dual averaging**으로 $\epsilon$ 튜닝
- Implicit acceptance (Detailed balance 유지)
- **현대 PPL의 기본**

---

## 🤔 생각해볼 문제

**문제 1** (기초): HMC의 step size가 너무 작으면? 너무 크면?

<details>
<summary>해설</summary>

**너무 작음** ($\epsilon \downarrow$):
- Acceptance → 1 (Leapfrog 오차 작음)
- 하지만 **trajectory 총 길이 감소** → mixing 느림
- 각 step이 무의미하게 작음

**너무 큼** ($\epsilon \uparrow$):
- Leapfrog 오차 폭발 → acceptance ↓
- 대부분 reject → chain 정체

Optimal: acceptance ~ 0.65 (이론). NUTS의 target default 0.8은 보수적 (divergence 회피).

Divergence (Stan 용어): $\Delta H$가 너무 커서 unstable → 해당 sample 버림. 많으면 posterior geometry 문제 시그널.

</details>

**문제 2** (심화): U-turn criterion이 **local**하지 않고 **global** (전체 trajectory)인 이유?

<details>
<summary>해설</summary>

Local U-turn (인접 step 간) 체크하면:
- 작은 oscillation에서도 premature 종료
- Long productive trajectories 놓침

Global U-turn (처음 $\theta_-$ vs 끝 $\theta_+$):
- 전체 trajectory의 **total displacement** 평가
- 작은 wiggle 무시, 진짜 "돌아옴" 감지

NUTS는 **sub-tree마다** U-turn 체크 + overall tree 체크 → 다층 검증.

수학적으로 이 구조가 detailed balance와 호환되도록 설계 (Hoffman-Gelman 2014의 기여).

</details>

**문제 3** (AI 연결): Stan의 **"divergent transition"** 경고는 무엇을 의미?

<details>
<summary>해설</summary>

Divergence: 특정 step에서 $|\Delta H| > 10^3$ (큰 threshold). 원인:

1. **Step size가 너무 큼** → step size adaptation 재실행 (target_accept ↑)
2. **Posterior geometry pathological**:
   - Funnel (hierarchical $\sigma$): bottom에서 curvature 극단 → non-centered reparam
   - Correlation 극단: 적절한 reparameterization
3. **Prior가 너무 diffuse**: posterior 지역이 작아 numerical issues

해결 (in order):
- `target_accept=0.99`
- 모델 재매개변수화 (`non-centered`)
- Prior tighten
- Num warmup 증가

이것이 현대 Bayesian DL workflow에서 흔히 만나는 "**modeling = computation**" 철학 — 모델 설계가 sampler의 geometry와 엮임.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Hamiltonian Monte Carlo](./03-hamiltonian-monte-carlo.md) | [📚 README로 돌아가기](../README.md) | [05. 수렴 진단 — R̂과 ESS ▶](./05-convergence-diagnostics.md) |

</div>
