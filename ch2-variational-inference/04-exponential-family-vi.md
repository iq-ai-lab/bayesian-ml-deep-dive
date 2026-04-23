# 04. Exponential Family와 자연매개변수 VI

## 🎯 핵심 질문

- **Exponential family** $p(x;\eta) = h(x)\exp(\eta^T T(x) - A(\eta))$의 구조는 왜 VI에 결정적인가?
- **Conjugate-exponential** 모델에서 CAVI가 왜 **닫힌형**으로 떨어지는가?
- **Natural gradient**는 일반 gradient와 어떻게 다르며 왜 더 효율적인가?
- **Stochastic VI** (Hoffman et al. 2013)가 어떻게 대규모 데이터를 처리하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**LDA**에 수천만 문서, **Bayesian topic model**에 스트리밍 데이터 — 전통 CAVI로는 메모리·속도 못 맞춤. SVI는 mini-batch로 ELBO gradient를 natural gradient와 결합해 대규모 Bayesian 모델을 가능케 한다. **ADVI**(Stan·PyMC 자동 VI)도 이 framework. BNN에서 **natural-gradient VI**(Khan et al. 2018)가 SGD보다 빠른 수렴을 보이는 이론적 근거.

---

## 📐 수학적 선행 조건

- [Ch1-03 Conjugate Priors](../ch1-bayesian-foundation/03-conjugate-priors.md): Exponential family
- [Ch2-03 CAVI](./03-mean-field-cavi.md): 좌표 업데이트
- [Information Geometry Deep Dive](https://github.com/iq-ai-lab/information-geometry-deep-dive): Fisher metric, natural gradient
- Legendre transform (convex conjugate)

---

## 📖 직관적 이해

### Exponential Family의 특별함

$p(x;\eta) = h(x)\exp(\eta^T T(x) - A(\eta))$:

- $\eta$: **자연매개변수**(natural parameter)
- $T(x)$: **충분통계량**(sufficient statistic)
- $A(\eta)$: log partition → $\nabla A(\eta) = \mathbb{E}[T(X)]$ (**moment parameter**)
- $h(x)$: base measure

**변환**: $\eta \leftrightarrow \mu = \nabla A(\eta)$ — Legendre duality.

### Conjugate-Exponential 조합

- **Prior** exponential family on $\theta$, natural param $\lambda$
- **Likelihood** exponential family on $x$, natural param $\eta(\theta)$
- 둘의 곱이 같은 exp family: posterior가 **업데이트된 natural param**

CAVI에서 이 구조가 있으면 **$\mathbb{E}_{q_{-i}}[\eta]$만 계산**해서 $q_i^*$의 자연매개변수가 직접 나옴.

### Natural Gradient

일반 gradient $\nabla_\phi \mathcal{L}$은 **parameter space geometry**를 고려 안 함. 자연 gradient는 **Fisher 정보 행렬로 precondition**:

$$\tilde\nabla = F^{-1}\nabla_\phi \mathcal{L}$$

Exp family에선 $F = \nabla^2 A(\eta)$이므로 **Hessian**과 연결.

### 요리 비유

Exp family = "재료 목록 $T(x)$와 조리 강도 $\eta$로만 표현되는 요리". 모든 요리 상태가 이 두 축으로 표현 → 조합·업데이트가 **덧셈**처럼 단순.

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Exponential Family

$$p(x|\eta) = h(x)\exp(\eta^T T(x) - A(\eta))$$

$\eta \in \Omega$ (natural param space), $A(\eta) = \log \int h(x)\exp(\eta^T T(x))dx$.

### 정의 4.2 — Mean Parameter & Legendre Duality

$$\mu := \mathbb{E}_{p(\cdot|\eta)}[T(X)] = \nabla A(\eta)$$

$A$가 strictly convex면 $\eta \mapsto \mu$는 **일대일**. 역사상(Legendre):

$$A^*(\mu) = \sup_\eta (\eta^T \mu - A(\eta))$$

### 정의 4.3 — Fisher Information

$$F(\eta) = \nabla^2 A(\eta) = \text{Cov}_{p(\cdot|\eta)}(T(X))$$

exp family에선 Hessian of log-partition.

### 정의 4.4 — Natural Gradient

Parameter $\phi$에 대해:
$$\tilde\nabla_\phi \mathcal{L} = F(\phi)^{-1}\nabla_\phi \mathcal{L}$$

Fisher-Rao metric에서의 steepest ascent.

---

## 🔬 정리와 증명

### 정리 4.1 — Conjugate-Exp CAVI의 닫힌형

**명제**: Likelihood $p(x|\theta)$와 prior $p(\theta|\lambda)$가 exp family conjugate이면, mean-field CAVI 업데이트:

$$\eta_i^{post} = \lambda_i + \sum_{n} \mathbb{E}_{q_{-i}}[T_i(\theta_{-i}, x_n)]$$

즉 **자연매개변수 공간에서 덧셈**.

**증명**: Ch1-03 정리 3.6의 CAVI 버전. Log-joint를 $\theta_i$에 대해 정리하면 exp family form이 나오고, 자연매개변수가 현재 $q_{-i}$에서의 $T$의 기댓값으로 정리됨. $\square$

### 정리 4.2 — Natural Gradient = Expected Euclidean Gradient

**명제**: Variational params $\phi$가 exp family의 자연매개변수이고 (fully-specified $q_\phi$가 exp family), ELBO gradient에 대해:

$$\tilde\nabla_\phi \mathcal{L} = F(\phi)^{-1}\nabla_\phi \mathcal{L}$$

가 **간결한 closed form**을 가짐 (Hoffman et al. 2013, Sec 2.4):

$$\tilde\nabla_\phi \mathcal{L}_{SVI} = \hat\phi^* - \phi$$

where $\hat\phi^*$는 현재 mini-batch로 계산한 natural param의 "target" (single-step CAVI analog).

**증명 스케치**: Exp family에서 $F(\phi) = \nabla^2 A(\phi)$. Variational ELBO의 gradient가 $\nabla^2 A(\phi) \cdot (\hat\phi^* - \phi)$ 형태가 됨 (after manipulation). Fisher로 precondition하면 $(\hat\phi^* - \phi)$만 남음. $\square$

### 정리 4.3 — Stochastic Variational Inference (SVI)

**알고리즘**: Global variational param $\phi$에 대해:

1. Sample mini-batch $B \subset \{1, \ldots, N\}$
2. Compute local $q_i$ (local updates) for $i \in B$
3. Compute "intermediate" global $\hat\phi_B = \lambda + \frac{N}{|B|}\sum_{i \in B} \mathbb{E}_{q_i}[T_i]$
4. Update $\phi^{(t+1)} = (1 - \rho_t)\phi^{(t)} + \rho_t \hat\phi_B$

step size $\rho_t$ must satisfy Robbins-Monro: $\sum\rho_t = \infty, \sum\rho_t^2 < \infty$.

**정리**: 이 업데이트가 **natural gradient stochastic ascent**에 해당하고, 적절한 조건 하 ELBO 국소 최대화로 수렴.

**증명**: Hoffman et al. 2013 (Stochastic Variational Inference, JMLR). $\square$

### 정리 4.4 — Moment Matching

**명제**: Forward KL $\text{KL}(p\|q)$를 minimize하면 (reverse와 다른 방향!), $q$의 moments $\mathbb{E}_q[T]$가 $p$의 moments와 매치:

$$\mathbb{E}_q[T] = \mathbb{E}_p[T] \quad \text{(moment matching)}$$

**증명**: 

$$\text{KL}(p\|q) = \mathbb{E}_p[\log p - \log q]$$

$q$가 exp family $q(\theta;\eta) = h\exp(\eta^T T - A(\eta))$이면:

$$\text{KL}(p\|q) = -\mathbb{E}_p[\eta^T T(\theta) - A(\eta)] + \text{const}$$

$\nabla_\eta \text{KL}(p\|q) = -\mathbb{E}_p[T] + \nabla A(\eta) = -\mathbb{E}_p[T] + \mathbb{E}_q[T] = 0$

⇒ $\mathbb{E}_q[T] = \mathbb{E}_p[T]$. $\square$

> **Expectation Propagation**(EP)이 이 원리로 작동 — forward KL minimization via moment matching.

### 예시 — Beta-Bernoulli의 Exp Family 관점

Bernoulli: $p(x|\theta) = \theta^x(1-\theta)^{1-x} = \exp(x\log\frac{\theta}{1-\theta} - (-\log(1-\theta)))$

⇒ $\eta = \log\frac{\theta}{1-\theta}$ (**logit**), $T(x) = x$, $A(\eta) = \log(1 + e^\eta)$.

Conjugate prior = Beta, exp family with natural params.

Posterior 자연매개변수 업데이트: $(\alpha, \beta) \to (\alpha + k, \beta + n - k)$ ← **단순 덧셈** (정리 4.1).

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import digamma

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# SVI for Bayesian Gaussian Mixture (from Ch2-03)
# ────────────────────────────────────────────────
# 대규모 mini-batch 시뮬레이션
N = 10_000
K = 3
true_means = np.array([-3.0, 0.0, 3.0])
z_true = rng.integers(0, K, size=N)
xs = rng.normal(true_means[z_true], 0.7)

alpha0, sigma0 = 1.0, 5.0

# Global variational params (natural param form)
# q(mu_k) = N(m_k, s_k²), q(pi) = Dir(α_k)
m_k = rng.standard_normal(K)
s_k_sq = np.ones(K)
alpha_k = np.ones(K) * alpha0

def svi_step(batch_x, m_k, s_k_sq, alpha_k, alpha0, sigma0, rho, N_total):
    B = len(batch_x)
    # Local: q(z_i) ∝ exp(E[log π_k] + E[log N(x_i; μ_k, 1)])
    E_log_pi = digamma(alpha_k) - digamma(alpha_k.sum())
    E_log_lik = -0.5 * (batch_x[:, None]**2 - 2 * batch_x[:, None] * m_k + (m_k**2 + s_k_sq))
    log_qz = E_log_pi + E_log_lik
    log_qz -= log_qz.max(axis=1, keepdims=True)
    q_z = np.exp(log_qz); q_z /= q_z.sum(axis=1, keepdims=True)
    
    # Intermediate "target" natural params (scale up by N/B)
    N_k_batch = q_z.sum(axis=0)
    scaled_N_k = N_total / B * N_k_batch
    scaled_sum_x = N_total / B * (q_z * batch_x[:, None]).sum(axis=0)
    
    # Target: s_k^(-2) = 1/sigma0^2 + scaled_N_k, m_k = s_k^2 * scaled_sum_x
    s_k_sq_target = 1.0 / (1/sigma0**2 + scaled_N_k)
    m_k_target = s_k_sq_target * scaled_sum_x
    alpha_k_target = alpha0 + scaled_N_k
    
    # Natural gradient step (convex combination)
    m_k = (1 - rho) * m_k + rho * m_k_target
    s_k_sq = (1 - rho) * s_k_sq + rho * s_k_sq_target
    alpha_k = (1 - rho) * alpha_k + rho * alpha_k_target
    return m_k, s_k_sq, alpha_k

B = 100
for t in range(500):
    batch_idx = rng.integers(0, N, size=B)
    rho = (t + 10)**(-0.7)
    m_k, s_k_sq, alpha_k = svi_step(xs[batch_idx], m_k, s_k_sq, alpha_k, alpha0, sigma0, rho, N)

print(f"SVI learned means: {np.sort(m_k)}")
print(f"True means:        {np.sort(true_means)}")

fig, ax = plt.subplots(figsize=(10, 4))
x_plot = np.linspace(-6, 6, 400)
ax.hist(xs, bins=80, density=True, alpha=0.3, label='data')
for k in range(K):
    weight = alpha_k[k] / alpha_k.sum()
    ax.plot(x_plot, weight * stats.norm(m_k[k], 0.7).pdf(x_plot),
            lw=2, label=f'comp {k}: μ={m_k[k]:.2f}')
ax.legend(); ax.grid(alpha=0.3)
ax.set_title(f'SVI Bayesian GMM on N={N} (mini-batch B={B})')
plt.tight_layout(); plt.savefig('svi_gmm.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Topic Models
Stochastic VI가 Wikipedia-scale LDA를 가능케 함 (Hoffman 2010 online LDA).

### ADVI (Automatic Differentiation VI)
Kucukelbir et al. 2017. Sample을 unconstrained space에서 + reparameterization + SGD on ELBO. Natural gradient option도 제공.

### Natural Gradient in BNN
Khan-Nielsen's Variational Online Newton (2018) — BNN에 natural gradient VI. SGD 대비 빠른 수렴.

### EP (Expectation Propagation)
Moment matching (정리 4.4) 기반. Gaussian process classification 등에서 사용.

### Conjugate-Exponential Observational Models
ChatGPT-scale은 아니지만 **structured probabilistic models**에서 여전히 강력. Stan·PyMC 내부.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Exp family family | 일반 NN likelihood는 아님 → 직접 적용 불가 |
| Conjugate 구조 | 비공액 시 $\mathbb{E}_{q_{-i}}[\log p]$ MC 필요 |
| Natural gradient 계산 | 고차원 Fisher 역행렬 비용 |
| Step size schedule | Robbins-Monro 적절한 $\rho_t$ 선택 |
| Global/local 구분 | 복잡 hierarchy에서 분리 어려움 |

**실무 팁**: PyMC/NumPyro의 SVI는 SVI + reparameterization의 hybrid. Pure natural gradient는 specialized libraries (TF-Probability, Pyro).

---

## 📌 핵심 정리

$$\boxed{p(x|\eta) = h(x)\exp(\eta^T T(x) - A(\eta))}$$

핵심:
- Conjugate-exp ⇒ 자연매개변수 공간에서 **덧셈 업데이트**
- **Natural gradient** = Fisher 기반 precondition → exp family에선 closed form
- **SVI** (Hoffman 2013): mini-batch + stochastic natural-gradient ascent
- $\text{KL}(p\|q)$ min = **moment matching** (EP의 기초)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Gaussian $\mathcal{N}(\mu, \sigma^2)$를 exp family로 쓸 때 natural param $\eta$와 sufficient statistic $T(x)$는?

<details>
<summary>해설</summary>

$p(x|\mu, \sigma^2) = \frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$

$= \frac{1}{\sqrt{2\pi}}\exp\left(\frac{\mu}{\sigma^2}x - \frac{1}{2\sigma^2}x^2 - \frac{\mu^2}{2\sigma^2} - \log\sigma\right)$

$\eta = (\mu/\sigma^2, -1/(2\sigma^2))$, $T(x) = (x, x^2)$, $A(\eta) = -\eta_1^2/(4\eta_2) - \frac{1}{2}\log(-2\eta_2)$.

Sufficient statistics는 **평균과 2차 모멘트** — Gaussian의 모든 정보를 요약.

</details>

**문제 2** (심화): SVI에서 step size $\rho_t = (t + \tau)^{-\kappa}$, $0.5 < \kappa \leq 1$ 요구. 왜?

<details>
<summary>해설</summary>

Robbins-Monro 조건:
- $\sum\rho_t = \infty$: 충분한 exploration
- $\sum\rho_t^2 < \infty$: gradient noise control

$\rho_t = t^{-\kappa}$의 경우 $\sum t^{-\kappa} = \infty$ iff $\kappa \leq 1$, $\sum t^{-2\kappa} < \infty$ iff $2\kappa > 1 \Leftrightarrow \kappa > 0.5$.

Hoffman et al. 권고: $\kappa \in (0.5, 1)$, 예: $\kappa = 0.7, \tau = 10$. $\kappa \to 1$이면 conservative, $\kappa \to 0.5$이면 aggressive.

</details>

**문제 3** (AI 연결): PyTorch의 Adam optimizer가 "adaptive preconditioning"을 하는데, 이것과 natural gradient는 어떻게 다른가?

<details>
<summary>해설</summary>

| | Adam | Natural gradient |
|---|---|---|
| Precondition | 과거 gradient의 running moment | Fisher 정보 |
| 기반 geometry | Ad-hoc | **Fisher-Rao Riemannian** |
| 이론적 justify | Heuristic | statistical manifold steepest ascent |
| 비용 | Cheap (element-wise) | Expensive (matrix inversion) |

Adam은 **diagonal approximation** of preconditioner — 일종의 Fisher diagonal 근사로 볼 수 있음. 현대 BNN에선 Adam이 더 보편적이지만, natural-gradient VI가 iteration 수 면에선 빠름.

**K-FAC** (Kronecker-Factored Approximate Curvature)는 block-diagonal Fisher로 중간 영역 — BNN 실전에서 사용.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Mean-Field VI와 CAVI](./03-mean-field-cavi.md) | [📚 README로 돌아가기](../README.md) | [05. Reparameterization Trick ▶](./05-reparameterization-trick.md) |

</div>
