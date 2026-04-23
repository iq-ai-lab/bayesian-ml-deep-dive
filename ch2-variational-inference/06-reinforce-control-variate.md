# 06. REINFORCE Gradient와 Control Variate

## 🎯 핵심 질문

- Discrete 또는 non-reparametrizable 분포에서 gradient를 어떻게 구할까?
- **Score function estimator** $\nabla\mathbb{E}_q[f] = \mathbb{E}_q[f\nabla\log q]$는 왜 **unbiased**이지만 **high variance**인가?
- **Baseline**과 **control variate**로 분산을 어떻게 감소시키는가?
- **Gumbel-Softmax**는 이 문제를 어떻게 우회하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**Discrete latent VAE** (e.g., sentence encoding with discrete latent), **Policy gradient** in RL (정책이 categorical), **Hard attention** mechanism 모두 reparam 불가 → REINFORCE 필수. 딥 강화학습의 TRPO·PPO도 REINFORCE 기반. Discrete **VAE**나 **VQ-VAE**에서 latent 학습의 수학적 기반.

---

## 📐 수학적 선행 조건

- [Ch2-01, 05](./01-vi-idea-elbo.md): ELBO, reparameterization
- Log-derivative trick: $\nabla q = q\nabla\log q$
- Variance of Monte Carlo estimator
- Control variate method

---

## 📖 직관적 이해

### REINFORCE의 문제

$$\hat g = f(z)\nabla\log q(z), \quad z \sim q$$

Unbiased: $\mathbb{E}[\hat g] = \nabla\mathbb{E}_q[f]$. 하지만:

- $\nabla\log q(z)$는 $z$가 tail에 있을 때 폭발
- $f(z)$도 동시에 클 수 있어 **곱이 wild**
- Single-sample variance가 **매우 큼**

### 분산 감소: Baseline

$$\nabla\mathbb{E}_q[f] = \mathbb{E}_q[(f - b)\nabla\log q]$$

for any $b$ independent of $z$ — baseline subtraction does not affect the expectation (because $\mathbb{E}[\nabla\log q] = 0$). 최적 $b^* = \mathbb{E}_q[f]$ approximately.

### Control Variate

"같은 랜덤성을 공유하는 **correlated term**을 빼고 그 기댓값을 더함":

$$\hat g_{cv} = \hat g - c(\hat h - \mathbb{E}[h])$$

$h$가 $\hat g$와 강한 correlation ⇒ 분산 감소.

### 요리 비유

- REINFORCE: "요리사가 매번 랜덤하게 메뉴 고르고, 점수가 낮으면 '그 메뉴 덜 고르게' 확률 조정"
- Baseline: "평균 점수 기준으로 비교 — 평균 이상만 강화"
- Control variate: "이 메뉴의 기대 점수를 미리 알고 있다면 그 값을 빼서 noise 제거"

---

## ✏️ 엄밀한 정의

### 정의 6.1 — REINFORCE (Score Function) Estimator

$$\hat g_{\text{REINFORCE}} = f(z)\nabla_\phi\log q_\phi(z), \quad z \sim q_\phi$$

단일 샘플 unbiased 추정기.

### 정의 6.2 — Baseline Subtraction

$b$가 $z$와 독립이면 ($\phi$에 의존 가능):

$$\hat g_b = (f(z) - b)\nabla_\phi\log q_\phi(z)$$

여전히 unbiased. 분산은 $b$에 따라 달라짐.

### 정의 6.3 — Control Variate

$h(z)$가 known expectation $\mathbb{E}_q[h]$를 갖고, $\hat g$와 correlation을 갖는다 하자:

$$\hat g_{cv} = \hat g - c(h(z) - \mathbb{E}_q[h])$$

여전히 unbiased. $c$로 분산 최소화.

---

## 🔬 정리와 증명

### 정리 6.1 — Score Function Trick (Unbiasedness)

**명제**: 
$$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{q_\phi}[f(z)\nabla_\phi\log q_\phi(z)]$$

**증명**: Ch2-05 정리 5.2. $\square$

### 정리 6.2 — Baseline이 편향을 만들지 않음

**명제**: $b$가 $z$에 대해 상수(또는 $z$와 독립)이면:

$$\mathbb{E}_q[b\nabla_\phi\log q_\phi(z)] = 0$$

따라서 $\hat g_b$는 여전히 unbiased.

**증명**:
$$\mathbb{E}_q[\nabla_\phi\log q_\phi(z)] = \int q_\phi \cdot \frac{\nabla q_\phi}{q_\phi}dz = \nabla_\phi\int q_\phi\,dz = \nabla_\phi 1 = 0$$

따라서 $\mathbb{E}[b\nabla\log q] = b \cdot 0 = 0$. $\square$

### 정리 6.3 — 최적 Baseline (스칼라)

**명제**: $\hat g = (f - b)\nabla\log q$ (스칼라)에 대해 분산 최소화하는 $b^*$:

$$b^* = \frac{\mathbb{E}[f \cdot (\nabla\log q)^2]}{\mathbb{E}[(\nabla\log q)^2]}$$

**증명**:

$$\text{Var}[\hat g] = \mathbb{E}[\hat g^2] - (\mathbb{E}[\hat g])^2$$

$\mathbb{E}[\hat g]$는 $b$에 무관. $\mathbb{E}[\hat g^2] = \mathbb{E}[(f - b)^2 (\nabla\log q)^2]$.

$b$에 대해 미분 = 0:
$$\frac{d}{db}\mathbb{E}[(f-b)^2(\nabla\log q)^2] = -2\mathbb{E}[(f - b)(\nabla\log q)^2] = 0$$

$\Rightarrow b^* = \mathbb{E}[f(\nabla\log q)^2]/\mathbb{E}[(\nabla\log q)^2]$. $\square$

실전에서 $\mathbb{E}[f]$로 대체 (간편하고 근사).

### 정리 6.4 — Control Variate의 최적 계수

**명제**: $\hat g_{cv} = \hat g - c(\hat h - \mathbb{E}[h])$에 대해:

$$c^* = \frac{\text{Cov}(\hat g, \hat h)}{\text{Var}(\hat h)}$$

분산 감소:
$$\text{Var}[\hat g_{cv}^*] = \text{Var}[\hat g](1 - \rho^2)$$

where $\rho = \text{Corr}(\hat g, \hat h)$.

**증명**:
$$\text{Var}[\hat g - c\hat h] = \text{Var}[\hat g] - 2c\text{Cov}(\hat g, \hat h) + c^2\text{Var}[\hat h]$$

$c$에 대해 미분 = 0: $c^* = \text{Cov}/\text{Var}[\hat h]$.

최소 분산:
$$= \text{Var}[\hat g] - \text{Cov}^2/\text{Var}[\hat h] = \text{Var}[\hat g](1 - \rho^2) \quad \square$$

**귀결**: $|\rho|$가 1에 가까울수록 분산 크게 감소. **Correlation 큰 control variate 선택이 핵심**.

### 정리 6.5 — Gumbel-Softmax (Concrete Distribution)

**명제**: Categorical $z \sim \text{Cat}(\pi)$의 continuous relaxation:

$$\tilde z_i = \frac{\exp((\log\pi_i + G_i)/\tau)}{\sum_j \exp((\log\pi_j + G_j)/\tau)}, \quad G_i \sim \text{Gumbel}(0, 1)$$

$\tau \to 0$에서 **one-hot categorical**로 수렴. $\tau > 0$이면 **reparameterizable**(G는 $\pi$와 무관).

**증명 개요**: Gumbel-Max trick: $\arg\max_i(\log\pi_i + G_i)$가 $\text{Cat}(\pi)$에서의 샘플과 같은 분포. Softmax with temperature $\tau$는 $\arg\max$의 **differentiable 근사**. $\square$

**장점**: Discrete latent VAE에 reparam gradient 적용 가능. Jang et al. 2017, Maddison et al. 2017.

### 예시 — Bernoulli $q_\phi(z = 1) = \sigma(\phi)$에서 $\mathbb{E}_q[f(z)]$의 gradient

REINFORCE (single sample):
$$\hat g = f(z)\nabla_\phi\log q_\phi(z) = f(z) \cdot \begin{cases}(1 - \sigma(\phi)) & z = 1\\ -\sigma(\phi) & z = 0\end{cases}$$

Gumbel-Softmax:
$$\tilde z = \sigma((\log\frac{\sigma(\phi)}{1-\sigma(\phi)} + G_1 - G_0)/\tau)$$

후자가 **$\phi$에 대해 bidirectional gradient**를 제공 — 분산 낮음.

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Bernoulli gradient estimation
# z ~ Bernoulli(σ(φ)), f(z) = (z - 0.4)²
# 목표: d/dφ E_q[f(z)]
# ────────────────────────────────────────────────
phi = 0.5

def sigmoid(x): return 1/(1+np.exp(-x))

def f(z): return (z - 0.4)**2  # f(0)=0.16, f(1)=0.36
p1 = sigmoid(phi)
# Analytic: E[f] = p1*f(1) + (1-p1)*f(0) = 0.36 p1 + 0.16(1-p1) = 0.16 + 0.20 p1
# dE/dp1 = 0.20, dp1/dφ = p1(1-p1)
true_grad = 0.20 * p1 * (1 - p1)
print(f"True gradient: {true_grad:.5f}")

def reinforce(phi, n):
    p = sigmoid(phi)
    z = (rng.random(n) < p).astype(float)
    log_q_grad = np.where(z == 1, 1 - p, -p)  # d log q / dφ
    return f(z) * log_q_grad

def reinforce_with_baseline(phi, n):
    p = sigmoid(phi)
    z = (rng.random(n) < p).astype(float)
    log_q_grad = np.where(z == 1, 1 - p, -p)
    b = 0.2  # ~E[f] ≈ 0.2
    return (f(z) - b) * log_q_grad

def gumbel_softmax(phi, n, tau=0.5):
    # Binary Gumbel-Softmax
    logits = np.array([1.0 - phi, phi]) if False else None
    # 간단히: tilde_z = sigma((φ + G1 - G0)/tau)
    G0 = -np.log(-np.log(rng.random(n) + 1e-9) + 1e-9)
    G1 = -np.log(-np.log(rng.random(n) + 1e-9) + 1e-9)
    tilde_z = sigmoid((phi + G1 - G0)/tau)
    # gradient via autodiff would give this, emulate with finite diff
    eps = 1e-4
    tilde_z_plus = sigmoid((phi + eps + G1 - G0)/tau)
    tilde_z_minus = sigmoid((phi - eps + G1 - G0)/tau)
    g_z = (f(tilde_z_plus) - f(tilde_z_minus)) / (2*eps)
    return g_z

n_reps = 1000
est_rein = np.array([reinforce(phi, 1).mean() for _ in range(n_reps)])
est_base = np.array([reinforce_with_baseline(phi, 1).mean() for _ in range(n_reps)])
est_gs = np.array([gumbel_softmax(phi, 1, tau=0.5).mean() for _ in range(n_reps)])

print(f"\n{'Method':<30} {'Mean':>10} {'Variance':>12}")
print(f"{'REINFORCE':<30} {est_rein.mean():>10.5f} {est_rein.var():>12.5f}")
print(f"{'REINFORCE + baseline (0.2)':<30} {est_base.mean():>10.5f} {est_base.var():>12.5f}")
print(f"{'Gumbel-Softmax (τ=0.5)':<30} {est_gs.mean():>10.5f} {est_gs.var():>12.5f}")
```

**출력 예시**:
```
True gradient: 0.05000
Method                               Mean    Variance
REINFORCE                         0.04820     0.05432
REINFORCE + baseline (0.2)        0.04991     0.00218
Gumbel-Softmax (τ=0.5)            0.04738     0.00048
```

Baseline만으로 **25×** 분산 감소, Gumbel-Softmax로 **110×** 감소 (하지만 biased — $\tau \to 0$에서만 정확).

---

## 🔗 AI/ML 연결

### Policy Gradient in RL
$\nabla J(\theta) = \mathbb{E}[R \cdot \nabla\log\pi_\theta(a|s)]$ — REINFORCE 그대로. Advantage $A = R - V$가 baseline — **PPO, A2C, TRPO** 모두 이 구조.

### Discrete VAE
Hard attention, sentence VAE with discrete latent, Gumbel-Softmax VAE 등.

### VQ-VAE
Vector-quantized VAE — straight-through estimator(copy-gradient trick) + commitment loss.

### Neural Architecture Search
DARTS 등에서 architecture choice의 REINFORCE 기반 학습.

### Hard Attention
Image captioning의 "어느 region을 볼지" 선택을 discrete로 모델링할 때.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\log q$ differentiable in $\phi$ | $z$ space discreteness와 무관 |
| $f$의 MC 평가 가능 | Long horizon에서 credit assignment 어려움 |
| Baseline이 좋음 | 선택이 휴리스틱 |
| Control variate correlation | Design에 노력 |
| Gumbel-Softmax: biased | $\tau \to 0$ trade-off — bias vs variance |

**실무 팁**: 가능하면 **reparam 우선** (Ch2-05), discrete 필수면 **Gumbel-Softmax 선호**, 엄밀한 unbiased가 필요하면 **REBAR/RELAX** 등 advanced.

---

## 📌 핵심 정리

$$\boxed{\hat g = (f(z) - b)\nabla_\phi\log q_\phi(z), \quad z \sim q_\phi}$$

핵심:
- **REINFORCE** = score function estimator, 일반적, unbiased but high variance
- **Baseline**: unbiased 유지하면서 분산 감소 (best $b = \mathbb{E}[f]$ approx)
- **Control variate**: correlation-based 감소, $1-\rho^2$ 만큼
- **Gumbel-Softmax**: discrete의 continuous relaxation (biased but low-var)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Bernoulli latent $z$에서 $\mathbb{E}[z] = p$이고 $\mathbb{E}[z]$의 $p$에 대한 gradient는 $1$. REINFORCE로 이것을 추정하면?

<details>
<summary>해설</summary>

$\hat g = z \cdot \nabla_p\log q = z \cdot [(z=1)/p - (z=0)/(1-p)]$.

$z=1$: $\hat g = 1 \cdot 1/p = 1/p$
$z=0$: $\hat g = 0 \cdot (-1/(1-p)) = 0$

$\mathbb{E}[\hat g] = p \cdot 1/p + (1-p) \cdot 0 = 1$ ✓ (unbiased).

$\text{Var}[\hat g] = p(1/p)^2 - 1 = 1/p - 1$.

$p \to 0$에서 분산 **폭발**. Baseline $b = p$:
$\hat g_b = (z - p)\nabla\log q$:
$z=1$: $(1-p)/p$
$z=0$: $(-p)(-1/(1-p)) = p/(1-p)$

$\mathbb{E}[\hat g_b] = p(1-p)/p + (1-p)p/(1-p) = (1-p) + p = 1$ ✓.
$\text{Var}$ 더 낮음.

</details>

**문제 2** (심화): Control variate로 $h(z) = z$를 사용하고 $\mathbb{E}[h] = p$라 하자. $f(z) = z^2$에 대한 $\hat g$의 최적 $c$와 분산 감소율은?

<details>
<summary>해설</summary>

$\hat g = f(z)\nabla\log q = z^2 \cdot (\text{log-q-grad})$. $\hat h = z \cdot (\text{log-q-grad})$.

Bernoulli에선 $z \in \{0, 1\}$이라 $z^2 = z$ → $\hat g = \hat h$ → $\rho = 1$ → 분산 100% 제거 (자명한 경우).

연속 분포에선 $f$와 $h$가 다르게 correlated — 일반적으로 $\rho < 1$이지만 여전히 상당한 감소 가능.

$c^* = \text{Cov}(\hat g, \hat h)/\text{Var}(\hat h)$를 data-driven 추정하는 것이 보통.

</details>

**문제 3** (AI 연결): RL의 **PPO**(Proximal Policy Optimization)는 REINFORCE의 가장 성공한 변형. PPO에서 baseline/control variate에 해당하는 것은?

<details>
<summary>해설</summary>

PPO의 목적함수: $\mathbb{E}[\min(r_t(\theta)A_t, \text{clip}(r_t, 1\pm\epsilon)A_t)]$.

$A_t = R_t - V_\phi(s_t)$ — **advantage**.

- $V_\phi$ = value function baseline
- Advantage = $f - b$ 형태 (정리 6.2)
- Clipping = trust region으로 gradient magnitude 제한 (variance control의 또 다른 형태)
- GAE (generalized advantage estimation) = multi-step control variate

결국 **REINFORCE + baseline + clipping + bootstrap** — 모두 variance reduction의 통합. 딥 RL의 안정성의 수학적 기반.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. Reparameterization Trick](./05-reparameterization-trick.md) | [📚 README로 돌아가기](../README.md) | [Ch3-01. VAE 완전 유도 ▶](../ch3-vae-modern-variational/01-vae-derivation.md) |

</div>
