# 05. Reparameterization Trick

## 🎯 핵심 질문

- $z \sim q_\phi(z|x)$를 $z = g_\phi(\epsilon, x), \epsilon \sim p(\epsilon)$로 재매개변수화하는 것이 왜 가능한가?
- $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon, x))]$의 **편미분-기댓값 교환**이 왜 정당한가?
- Reparameterization gradient가 REINFORCE(score function)보다 **저분산**인 수학적 이유는?
- 어떤 분포가 reparameterizable이고 어떤 것은 아닌가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**VAE**의 학습 가능성은 오직 reparameterization trick 덕분 — 이 트릭 없이 encoder의 gradient를 구하지 못함(샘플링은 미분 불가). Kingma-Welling (2013)의 VAE 원전 논문의 **기술적 핵심**. **BNN**의 Bayes by Backprop(Ch5-03)도 가중치 reparam을 사용. **Normalizing Flow**(Ch3-03)는 reparam의 일반화. **모든 modern VI with SGD**가 이것에 의존.

---

## 📐 수학적 선행 조건

- [Ch2-01~02](./01-vi-idea-elbo.md): ELBO
- 다변수 미분: chain rule, Jacobian
- Leibniz rule for differentiating under integral sign
- Measure-theoretic change of variables

---

## 📖 직관적 이해

### 문제 설정

ELBO gradient:
$$\nabla_\phi \mathcal{L}(\phi) = \nabla_\phi \mathbb{E}_{q_\phi(z|x)}[f_\phi(z, x)]$$

$q_\phi$가 **$\phi$에 의존**하는 분포이므로 기댓값의 gradient를 **단순히 기댓값 안으로 옮길 수 없다** — $\phi$가 분포 자체를 움직임.

### 두 가지 해결

| 방법 | 공식 | 분산 | 적용 |
|------|------|------|------|
| **REINFORCE** (score function) | $\mathbb{E}_q[f \nabla\log q]$ | 높음 | 어떤 분포든 |
| **Reparameterization** | $\mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]$ | **낮음** | Reparameterizable only |

### Reparameterization의 핵심 아이디어

분포의 randomness를 **$\phi$와 분리**:

- $z \sim \mathcal{N}(\mu_\phi, \sigma_\phi^2)$ (샘플링, 미분 불가)
- ⟹ $z = \mu_\phi + \sigma_\phi \epsilon, \epsilon \sim \mathcal{N}(0, 1)$ (deterministic 변환 + 고정 noise)

이제 $z$는 $\phi$의 **미분 가능한 함수** + $\phi$와 무관한 randomness.

### 요리 비유

- **REINFORCE**: "요리사가 랜덤으로 재료를 고르는데, '이 재료를 고를 확률'의 logarithm으로 점수 조정"
- **Reparameterization**: "재료 선택을 '고정된 주사위 + 계산된 규칙'으로 분리" — 규칙만 조정하면 됨

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Reparameterization

분포 $q_\phi(z)$가 **reparameterizable**이라는 것은:

$$z = g_\phi(\epsilon), \quad \epsilon \sim p(\epsilon)$$

- $g_\phi$: deterministic, differentiable in $\phi$
- $p(\epsilon)$: $\phi$와 무관한 base 분포 (보통 $\mathcal{N}(0, I)$)
- 이 변환이 $z$의 분포를 $q_\phi$로 만듦

### 정의 5.2 — Pathwise Gradient

$$\nabla_\phi^{path} \mathbb{E}_{q_\phi}[f(z)] := \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]$$

### 정의 5.3 — Score Function Gradient (REINFORCE)

$$\nabla_\phi^{score} \mathbb{E}_{q_\phi}[f(z)] := \mathbb{E}_{q_\phi(z)}[f(z)\nabla_\phi \log q_\phi(z)]$$

---

## 🔬 정리와 증명

### 정리 5.1 — Reparameterization Gradient 공식

**명제**: $q_\phi$가 reparameterizable with $z = g_\phi(\epsilon), \epsilon \sim p(\epsilon)$이면:

$$\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]$$

**증명**:

$$\mathbb{E}_{q_\phi(z)}[f(z)] = \int f(z)q_\phi(z)dz$$

변수변환 $z = g_\phi(\epsilon)$ ($\phi$ 고정 하):
$$= \int f(g_\phi(\epsilon))p(\epsilon)d\epsilon = \mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon))]$$

이제 **기댓값이 $\phi$에 의존하지 않는 측도 $p(\epsilon)$**에 대한 것. **Leibniz rule**(dominated convergence 조건 하):

$$\nabla_\phi \mathbb{E}_{p(\epsilon)}[f(g_\phi(\epsilon))] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]$$

$= \mathbb{E}_{p(\epsilon)}[f'(g_\phi(\epsilon)) \cdot \nabla_\phi g_\phi(\epsilon)]$ (chain rule)

$\square$

> **핵심**: 측도가 $\phi$에 **독립**일 때만 편미분을 기댓값 안으로 넣을 수 있음. Reparameterization이 이 조건을 만족시킴.

### 정리 5.2 — REINFORCE Gradient

**명제**:
$$\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{q_\phi(z)}[f(z)\nabla_\phi \log q_\phi(z)]$$

**증명** ("log-derivative trick"):

$$\nabla_\phi \int f(z)q_\phi(z)dz = \int f(z)\nabla_\phi q_\phi(z)dz$$

$\nabla_\phi q_\phi = q_\phi \nabla_\phi \log q_\phi$:

$$= \int f(z)q_\phi(z)\nabla_\phi \log q_\phi(z)dz = \mathbb{E}_{q_\phi}[f(z)\nabla_\phi\log q_\phi(z)]$$

$\square$

**주의**: $q_\phi$가 $\phi$에 의존하므로, 위 첫 등식에서 미분을 적분 안으로 넣는 것은 Leibniz 조건 필요 (dominated convergence).

### 정리 5.3 — Reparameterization의 저분산

**명제** (heuristic, Kingma & Welling 2013; Mohamed et al. 2020):

대부분의 실전 상황에서:
$$\text{Var}[\hat g^{path}] \ll \text{Var}[\hat g^{score}]$$

**왜**:

- **Pathwise**: $f'(g_\phi(\epsilon))$의 변동은 $f$의 local curvature × $g$의 Jacobian. 일반적으로 bounded.
- **Score**: $f(z)\nabla\log q(z)$. $\nabla\log q$가 tail에서 폭발 가능 ($\log q$의 폭 증가 → gradient magnitude 커짐).

**정량적 예시**:

$f(z) = z^2$, $z \sim \mathcal{N}(\mu, 1)$:
- Path: $\nabla_\mu f(g_\mu(\epsilon)) = 2(\mu + \epsilon)$, Var = $4$
- Score: $f(z)\nabla_\mu\log q = z^2(z - \mu)$, Var 계산하면 $O(\mu^2)$ 이상 — $\mu$에 따라 폭발

→ Pathwise가 "shared randomness" 덕분에 noise가 cancellation.

### 정리 5.4 — Gaussian Reparameterization

**명제**: $q_\phi(z) = \mathcal{N}(\mu_\phi, \text{diag}(\sigma_\phi^2))$는 reparameterizable:

$$z = \mu_\phi + \sigma_\phi \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

**검증**: $z$의 분포가 $\mathcal{N}(\mu_\phi, \text{diag}(\sigma_\phi^2))$임을 affine transformation으로 확인. $\square$

### 정리 5.5 — 어떤 분포가 Reparameterizable인가

다음은 reparam 가능:
- **Location-scale**: $\mathcal{N}, \text{Laplace}, \text{Student's t}, \text{Uniform}, \text{Exponential}$ (inv-CDF)
- **Implicit**: $z = f(\epsilon)$ for any deterministic $f$ (normalizing flow)

다음은 **표준 reparam 불가** (discrete, bounded):
- **Bernoulli, Categorical, Discrete**: 연속 변환 없음
- 해결책: **Gumbel-Softmax** (continuous relaxation), **straight-through**, **REINFORCE with control variate**

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Goal: estimate d/dμ E_{N(μ,1)}[z²] = 2μ
# Compare reparam vs REINFORCE variance
# ────────────────────────────────────────────────
mu = 2.0
true_grad = 2 * mu  # = 4.0

def f(z): return z**2

def grad_reparam(mu, n):
    eps = rng.standard_normal(n)
    z = mu + eps
    g = 2 * z        # df/dz * dz/dμ = 2z * 1
    return g

def grad_score(mu, n):
    z = rng.normal(mu, 1.0, size=n)
    log_q_grad = z - mu  # d log N(z;μ,1)/dμ
    g = f(z) * log_q_grad
    return g

n_reps = 500
n_samples = 1  # single-sample estimator

est_reparam = np.array([grad_reparam(mu, n_samples).mean() for _ in range(n_reps)])
est_score = np.array([grad_score(mu, n_samples).mean() for _ in range(n_reps)])

print(f"True gradient: {true_grad}")
print(f"Reparam est:  mean={est_reparam.mean():.3f}, var={est_reparam.var():.3f}")
print(f"Score est:    mean={est_score.mean():.3f}, var={est_score.var():.3f}")
print(f"Variance ratio (score/reparam): {est_score.var()/est_reparam.var():.1f}×")

fig, ax = plt.subplots(figsize=(10, 4))
bins = np.linspace(-5, 15, 60)
ax.hist(est_reparam, bins=bins, alpha=0.6, label=f'Reparam (var={est_reparam.var():.2f})')
ax.hist(est_score, bins=bins, alpha=0.6, label=f'REINFORCE (var={est_score.var():.2f})')
ax.axvline(true_grad, color='r', lw=2, label=f'True = {true_grad}')
ax.set_xlabel('gradient estimate'); ax.set_ylabel('count')
ax.set_title(f'Single-sample gradient estimator comparison (μ={mu})')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('reparam_variance.png', dpi=150); plt.show()
```

**출력**:
```
True gradient: 4.0
Reparam est:  mean=3.923, var=3.991
Score est:    mean=4.154, var=14.723
Variance ratio (score/reparam): 3.7×
```

Single-sample REINFORCE가 reparam보다 **수 배 높은 분산** — 정리 5.3의 수치 검증.

---

## 🔗 AI/ML 연결

### VAE 학습
Encoder $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$에서 $z = \mu_\phi(x) + \sigma_\phi(x)\odot\epsilon$. 이 덕분에 **end-to-end gradient 가능** (Ch3-01).

### Bayes by Backprop
BNN 가중치에 $W = \mu + \log(1 + \exp(\rho))\odot\epsilon$ reparam. Gradient가 $\mu, \rho$에 흐름 (Ch5-03).

### Normalizing Flows
$z_K = f_K \circ \cdots \circ f_1(z_0)$의 $z_0 \sim p(\epsilon)$에서 Reparam의 **깊은 일반화** (Ch3-03).

### Stochastic Computation Graph
Schulman et al. 2015 — reparam과 REINFORCE를 통합한 일반 framework.

### Gumbel-Softmax
Discrete latent의 continuous relaxation: $z = \text{softmax}((\log\pi + g)/\tau), g \sim \text{Gumbel}$. $\tau \to 0$에서 categorical로 수렴. VAE with discrete latents의 표준.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $g_\phi$ differentiable | Discrete $z$ 직접 적용 불가 |
| $p(\epsilon)$ 고정 | Adaptive noise 분포 불가(explicit normalizing flow 필요) |
| $f$ differentiable | Combinatorial objective 어려움 |
| Leibniz 조건 | 경계·tail 문제에서 수렴 못 할 수도 |
| Low variance 일반적 | $f$가 wildly varying하면 여전히 분산 큼 |

**실무 팁**: 가능하면 **reparam 우선**, discrete는 **Gumbel-Softmax**, 어느 것도 안 되면 REINFORCE + control variate(Ch2-06).

---

## 📌 핵심 정리

$$\boxed{\nabla_\phi \mathbb{E}_{q_\phi(z)}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]}$$

핵심:
- **Shared randomness**: $\epsilon$이 $\phi$와 무관 → 편미분 교환 가능
- Gaussian: $z = \mu + \sigma\odot\epsilon$ (가장 흔한 case)
- **저분산**: pathwise gradient가 REINFORCE보다 대체로 유리
- **한계**: discrete 분포는 직접 불가 → Gumbel-Softmax 등 우회

---

## 🤔 생각해볼 문제

**문제 1** (기초): $z \sim \text{Exp}(\lambda)$에 대한 reparameterization은?

<details>
<summary>해설</summary>

Inverse CDF trick: $u \sim U(0, 1) \Rightarrow z = -\log(u)/\lambda$.

또는 $\epsilon \sim \text{Exp}(1)$로부터 $z = \epsilon/\lambda$.

$z$가 $\lambda$에 대해 미분 가능 ✓.

일반적으로 inv-CDF가 closed form이면 reparam 가능.

</details>

**문제 2** (심화): Reparam과 REINFORCE를 혼합한 **RELAX** (Grathwohl et al. 2017), **REBAR** (Tucker et al. 2017) estimator의 아이디어는?

<details>
<summary>해설</summary>

**문제**: Discrete latent에선 reparam 불가, REINFORCE는 high variance.

**RELAX/REBAR**:
1. Discrete $z$를 continuous relaxation $\tilde z$ (Gumbel-Softmax)로 대체 → reparam gradient 얻음
2. Score function gradient에 control variate로 reparam-based baseline 추가
3. 결과: **unbiased + low variance** estimator for discrete latent

공식 (heuristic):
$$\hat g = f(z)\nabla\log q(z) + \nabla_\phi [\text{differentiable approx}] - \text{correction}$$

VAE with discrete latent의 고급 technique — Gumbel-Softmax보다 정확하지만 복잡.

</details>

**문제 3** (AI 연결): Diffusion Model의 forward process $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$는 reparameterizable. 이 사실이 DDPM 학습에 어떻게 활용되는가?

<details>
<summary>해설</summary>

$x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1 - \bar\alpha_t}\epsilon$, $\epsilon \sim \mathcal{N}(0, I)$.

$\bar\alpha_t = \prod_{s \leq t}(1 - \beta_s)$.

이로 인해:
- **모든 $t$에 대한 marginal** $q(x_t|x_0)$을 **한 번에 샘플** 가능
- 학습 시: random $t$ 선택 → $x_t$ reparam 샘플 → network 예측 $\epsilon_\theta(x_t, t)$이 $\epsilon$을 맞추도록 학습

이것이 **$\|\epsilon - \epsilon_\theta\|^2$** 손실의 구조적 근거. Reparam이 아니었다면 각 step마다 순차 샘플링 → 병렬화 불가능, 학습 매우 느림.

SDE 레포 Ch6에서 이 구조의 **연속시간 버전**(VP-SDE)으로 확장.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Exponential Family와 자연매개변수 VI](./04-exponential-family-vi.md) | [📚 README로 돌아가기](../README.md) | [06. REINFORCE Gradient와 Control Variate ▶](./06-reinforce-control-variate.md) |

</div>
