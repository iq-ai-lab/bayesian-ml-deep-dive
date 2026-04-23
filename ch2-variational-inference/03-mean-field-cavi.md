# 03. Mean-Field VI와 CAVI

## 🎯 핵심 질문

- **Mean-field 가정** $q(\theta) = \prod_i q_i(\theta_i)$이 왜 inference를 단순화하는가?
- 좌표 업데이트 공식 $\log q_i^*(\theta_i) = \mathbb{E}_{q_{-i}}[\log p(x, \theta)] + \text{const}$는 어디서 오는가?
- **CAVI**(Coordinate Ascent VI) 알고리즘은 ELBO를 **단조 증가**시키는가?
- Mean-field의 **한계**(correlation 무시, underestimate variance)는?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**LDA**(Latent Dirichlet Allocation)의 표준 추론이 mean-field CAVI. **Bayesian Mixture Model**의 posterior, **Bayesian Probabilistic Matrix Factorization**, **Hidden Markov Model**의 Bayesian 추론 모두 mean-field CAVI로 시작. Conjugate-exponential 모델에서 **닫힌형 좌표 업데이트**가 가능해 code 한 줄에 구현 가능. 이해 없이는 PyMC, Edward 같은 PPL 내부 동작을 모른다.

---

## 📐 수학적 선행 조건

- [Ch2-01, 02](./01-vi-idea-elbo.md): VI 기초, ELBO 분해
- [Ch1-03 Conjugate Priors](../ch1-bayesian-foundation/03-conjugate-priors.md): Exponential family 구조
- Calculus of variations: functional derivative
- Lagrange multiplier, Euler-Lagrange equation

---

## 📖 직관적 이해

### Mean-field 가정

전체 posterior를 **곱 형태**로 분해:

$$q(\theta_1, \ldots, \theta_d) = \prod_i q_i(\theta_i)$$

모든 파라미터가 **독립**이라 가정. Posterior의 correlation은 완전히 무시.

**장점**:
- 각 $q_i$ 최적화가 다른 $j \neq i$의 현재 값에 **의존만** → 좌표별 닫힌형 업데이트
- Implementation 간단, conjugate-exp에선 해석해

**단점**:
- Posterior의 correlation 구조 완전 무시 → **분산 과소추정**
- Multimodal posterior의 한 mode만 잡음

### CAVI 알고리즘

**Coordinate Ascent VI**:

```
1. q_i 초기화
2. Repeat until ELBO 수렴:
   For i = 1, ..., d:
      q_i^* = argmax_{q_i} L(q)  ← 다른 j를 고정
3. Return q = ∏ q_i
```

각 step이 ELBO를 (안전하게) 증가 → **수렴 보장**(local optimum).

### 요리 비유

전체 부엌이 복잡. Mean-field = "재료별로 독립적으로 최적화": 야채 담당 · 고기 담당 · 소스 담당 각자 최선. 서로 간 coordination은 못 보지만 각자는 closed-form으로 답 가능.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Mean-Field Family

$\theta = (\theta_1, \ldots, \theta_d)$에 대해:

$$\mathcal{Q}_{MF} = \left\{q(\theta) = \prod_{i=1}^d q_i(\theta_i)\right\}$$

각 $q_i$는 자유로운 1차원 분포 — parametric 형태 고정 안 함(자연 mean-field) 또는 고정(fixed-form MF).

### 정의 3.2 — Free-Form vs Fixed-Form

- **Free-form MF**: $q_i$의 형태를 최적화로 결정 → CAVI가 자연스럽게 유도 (정리 3.1)
- **Fixed-form MF**: $q_i$를 특정 family(e.g., Gaussian)로 고정 → 파라미터만 최적화 → gradient-based

### 정의 3.3 — CAVI 업데이트

$i$번째 factor의 최적 업데이트:

$$\log q_i^*(\theta_i) = \mathbb{E}_{q_{-i}}[\log p(x, \theta)] + \text{const}$$

$\mathbb{E}_{q_{-i}}$는 **$\theta_i$ 이외 모든 변수에 대한 현재 $q$에서의 기댓값**.

---

## 🔬 정리와 증명

### 정리 3.1 — CAVI 좌표 업데이트 공식

**명제**: Mean-field family $q = \prod_j q_j$에서 $q_i$에 대해 ELBO를 최대화하면:

$$\log q_i^*(\theta_i) = \mathbb{E}_{q_{-i}}[\log p(x, \theta)] + \text{const}$$

**증명** (functional derivative / Lagrange):

ELBO를 $q_i$만의 함수로 정리 (분해 (3) 사용):

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] - \sum_j \mathbb{E}_{q_j}[\log q_j]$$

$q_i$에 대한 부분만 분리 (다른 $j$ 고정):

$$\mathcal{L}(q_i) = \mathbb{E}_{q_i}\left[\mathbb{E}_{q_{-i}}[\log p(x, \theta)]\right] - \mathbb{E}_{q_i}[\log q_i] + C$$

(C는 $q_i$에 무관한 상수.)

$\tilde p_i(\theta_i) := \exp(\mathbb{E}_{q_{-i}}[\log p(x, \theta)])$로 놓으면 (정규화 전):

$$\mathcal{L}(q_i) = \int q_i(\theta_i)\log \tilde p_i(\theta_i)\,d\theta_i - \int q_i\log q_i\,d\theta_i + C$$

$$= -\text{KL}(q_i \| \tilde p_i / Z) + \log Z + C$$

where $Z = \int \tilde p_i\,d\theta_i$.

ELBO 최대 iff $q_i = \tilde p_i / Z$, 즉:

$$q_i^*(\theta_i) \propto \exp(\mathbb{E}_{q_{-i}}[\log p(x, \theta)])$$

로그를 취하면 공식. $\square$

### 정리 3.2 — CAVI는 ELBO 단조 증가

**명제**: 각 CAVI 업데이트는 $\mathcal{L}$를 감소시키지 않는다:

$$\mathcal{L}(q^{(t+1)}) \geq \mathcal{L}(q^{(t)})$$

**증명**:

업데이트 전 $q^{(t)}$에서 $\mathcal{L}(q^{(t)}) = f(q_i^{(t)}, q_{-i}^{(t)})$. $q_{-i}$ 고정 하 $f$의 $q_i$ 최대화자가 $q_i^*$:

$$\mathcal{L}(q_i^*, q_{-i}^{(t)}) = \max_{q_i} f(q_i, q_{-i}^{(t)}) \geq f(q_i^{(t)}, q_{-i}^{(t)}) = \mathcal{L}(q^{(t)})$$

$\square$

**귀결**: $\mathcal{L}$가 위로 유계($\leq \log p(x)$)이므로 **수렴 보장**. 단, **local** optimum일 수 있음.

### 정리 3.3 — Exponential Family에서의 Closed Form

**명제**: Conjugate-exponential 모델에서 **완전조건부** $p(\theta_i|\theta_{-i}, x)$가 exponential family이고 자연매개변수 $\eta_i$가 $(\theta_{-i}, x)$의 함수이면:

$$q_i^*(\theta_i) \in \text{same exp family with natural parameter } \mathbb{E}_{q_{-i}}[\eta_i(\theta_{-i}, x)]$$

즉 **Gibbs sampler의 조건부 분포의 기댓값-기반 analog**.

**증명 스케치**: 

$\log q_i^*(\theta_i) = \mathbb{E}_{q_{-i}}[\log p(x, \theta)] + \text{const}$.

$\log p(x, \theta) = \log p(\theta_i|\theta_{-i}, x) + \log p(\theta_{-i}, x)$.

$\theta_i$에 대한 부분: $\log p(\theta_i|\theta_{-i}, x) = \eta_i(\theta_{-i}, x)^T T(\theta_i) - A(\eta_i) + h(\theta_i)$.

$\mathbb{E}_{q_{-i}}$ 취하면:

$$\log q_i^* = \mathbb{E}[\eta_i]^T T(\theta_i) - \mathbb{E}[A(\eta_i)] + h(\theta_i) + \text{const}$$

$-\mathbb{E}[A(\eta_i)]$는 $\theta_i$에 무관:

$$\log q_i^* = \mathbb{E}[\eta_i]^T T(\theta_i) + h(\theta_i) + \text{const}$$

같은 exp family, 자연매개변수 $\mathbb{E}[\eta_i]$. $\square$

> **실전 응용**: LDA에서 각 topic 분포 · document-topic 분포 · word assignment의 CAVI 업데이트가 모두 Dirichlet/Categorical exponential family 덕분에 닫힌형.

### 정리 3.4 — Mean-Field의 Posterior 분산 과소추정

**명제**: True posterior $p(\theta|x)$에 correlation이 있으면 mean-field $q^*$의 marginal variance는 **항상 참 marginal variance보다 작거나 같다**:

$$\text{Var}_{q^*}(\theta_i) \leq \text{Var}_p(\theta_i) \text{ typically}$$

(strict inequality when correlation nonzero)

**증명 스케치**: Reverse KL은 **mode-seeking** — posterior의 한 부분에 집중하는 경향. 특히 2D Gaussian posterior $\mathcal{N}(0, \Sigma)$에 대해 $q = \mathcal{N}(\mu, \text{diag}(\sigma^2))$의 최적 $\sigma_i^2 = 1/[\Sigma^{-1}]_{ii}$로, $\Sigma_{ii}$가 아닌 **precision의 역수의 diagonal**. Precision의 diagonal은 covariance의 diagonal보다 크므로 $\sigma_i^2 \leq \Sigma_{ii}$. $\square$

### 예시 — Bayesian Mixture of Gaussians (CAVI)

$x_i | z_i = k \sim \mathcal{N}(\mu_k, 1)$, $z_i \sim \text{Cat}(\pi)$, $\pi \sim \text{Dir}(\alpha)$, $\mu_k \sim \mathcal{N}(0, \sigma_0^2)$.

Mean-field $q(\mu, \pi, z) = q(\pi)\prod_k q(\mu_k)\prod_i q(z_i)$.

CAVI 업데이트 (Blei-Jordan 2006):
- $q(z_i = k) \propto \exp(\mathbb{E}[\log\pi_k] + \mathbb{E}[\log\mathcal{N}(x_i; \mu_k, 1)])$
- $q(\mu_k) = \mathcal{N}(m_k, s_k^2)$ with weighted statistics
- $q(\pi) = \text{Dir}(\alpha + \sum_i q(z_i))$

모두 닫힌형.

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Mixture of 2 Gaussians via CAVI
# ────────────────────────────────────────────────
N, K = 200, 2
x_true_means = np.array([-2.0, 2.0])
z_true = rng.integers(0, K, size=N)
xs = rng.normal(x_true_means[z_true], 1.0)

# Prior
alpha0 = 1.0
sigma0 = 5.0

# Init q(z) randomly
q_z = rng.dirichlet(np.ones(K), size=N)  # N × K

def cavi_step(xs, q_z, alpha0, sigma0):
    N, K = q_z.shape
    # q(mu_k) = N(m_k, s_k^2)
    #   s_k^(-2) = 1/sigma0^2 + sum q_z[:, k]
    #   m_k = s_k^2 * sum q_z[:, k] * x_i
    N_k = q_z.sum(axis=0)
    s_k_sq = 1.0 / (1/sigma0**2 + N_k)
    m_k = s_k_sq * (q_z * xs[:, None]).sum(axis=0)
    # E[mu_k] = m_k, E[mu_k^2] = m_k^2 + s_k_sq
    
    # q(pi) = Dir(alpha_k), alpha_k = alpha0 + N_k
    alpha_k = alpha0 + N_k
    # E[log pi_k] = digamma(alpha_k) - digamma(sum alpha_k)
    from scipy.special import digamma
    E_log_pi = digamma(alpha_k) - digamma(alpha_k.sum())
    
    # q(z_i = k) ∝ exp(E[log pi_k] + E[log N(x_i; mu_k, 1)])
    # E[log N(x_i; mu_k, 1)] = -0.5*(x_i^2 - 2 x_i E[mu_k] + E[mu_k^2]) + const
    E_log_lik = -0.5 * (xs[:, None]**2
                        - 2 * xs[:, None] * m_k
                        + (m_k**2 + s_k_sq))
    log_qz = E_log_pi + E_log_lik
    log_qz -= log_qz.max(axis=1, keepdims=True)
    q_z = np.exp(log_qz); q_z /= q_z.sum(axis=1, keepdims=True)
    return q_z, m_k, s_k_sq, alpha_k

# ELBO
def compute_elbo(xs, q_z, m_k, s_k_sq, alpha_k, alpha0, sigma0):
    from scipy.special import digamma, gammaln
    N, K = q_z.shape
    N_k = q_z.sum(axis=0)
    E_log_pi = digamma(alpha_k) - digamma(alpha_k.sum())
    # E_q[log p(x, z, μ, π)]
    E_log_lik = -0.5*np.log(2*np.pi) - 0.5*(xs[:,None]**2 - 2*xs[:,None]*m_k + (m_k**2 + s_k_sq))
    term_x_given_z = (q_z * E_log_lik).sum()
    term_z = (q_z * E_log_pi).sum()
    term_pi = (alpha0 - 1) * E_log_pi.sum()
    term_mu = -0.5 * np.sum((m_k**2 + s_k_sq) / sigma0**2) - K*0.5*np.log(2*np.pi*sigma0**2)
    # -E_q[log q]
    H_z = -(q_z * np.log(q_z + 1e-30)).sum()
    H_mu = 0.5 * np.sum(np.log(2*np.pi*np.e*s_k_sq))
    H_pi = (gammaln(alpha_k).sum() - gammaln(alpha_k.sum())
            - np.sum((alpha_k - 1)*(digamma(alpha_k) - digamma(alpha_k.sum()))))
    return term_x_given_z + term_z + term_pi + term_mu + H_z + H_mu + H_pi

elbo_traj = []
for it in range(30):
    q_z, m_k, s_k_sq, alpha_k = cavi_step(xs, q_z, alpha0, sigma0)
    elbo_traj.append(compute_elbo(xs, q_z, m_k, s_k_sq, alpha_k, alpha0, sigma0))

print(f"Learned means: {np.sort(m_k)}")
print(f"True means:    {np.sort(x_true_means)}")
print(f"ELBO trajectory increased monotonically: {np.all(np.diff(elbo_traj) >= -1e-6)}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(elbo_traj, 'o-')
axes[0].set_xlabel('CAVI iter'); axes[0].set_ylabel('ELBO')
axes[0].set_title('CAVI는 ELBO를 단조 증가시킴 (정리 3.2)')
axes[0].grid(alpha=0.3)

x_plot = np.linspace(-6, 6, 400)
axes[1].hist(xs, bins=40, density=True, alpha=0.3, label='data')
for k in range(K):
    axes[1].plot(x_plot, stats.norm(m_k[k], 1.0).pdf(x_plot)*alpha_k[k]/alpha_k.sum(),
                 lw=2, label=f'component {k}: μ={m_k[k]:.2f}')
axes[1].legend(); axes[1].set_title('학습된 Mixture')
axes[1].grid(alpha=0.3)
plt.tight_layout(); plt.savefig('cavi_mixture.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### LDA (Latent Dirichlet Allocation)
Blei, Ng, Jordan (2003). Topic model 추론을 CAVI로. Dirichlet-Multinomial conjugate 구조가 핵심.

### Probabilistic Matrix Factorization
User/item latent factor를 mean-field CAVI로 추론.

### HMM Bayesian Inference
Transition/emission matrix에 Dirichlet prior → CAVI로 state posterior + parameter posterior 동시.

### VAE와의 차이
VAE는 **amortized**(NN encoder) + **fixed-form** Gaussian. CAVI는 non-amortized + 자유 form. 다른 trade-off.

### Structured VI (beyond mean-field)
Correlation을 보존하는 full-rank or low-rank approximation — Tran et al. 2015, 2016.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Factorized $q$ | Correlation 무시 → variance 과소추정 |
| Conjugate-exponential 구조 | 비공액 시 $\mathbb{E}_{q_{-i}}$ 계산 어려움 |
| CAVI 수렴 | Local optimum (global 보장 없음) |
| $\theta_i$ 분리 가능 | Tight coupling 모델(e.g., global parameter) 문제 |

**실무 팁**: Mean-field는 posterior **mean**은 잘 추정하지만 **uncertainty**는 underestimate. Downstream application에 따라 주의.

---

## 📌 핵심 정리

$$\boxed{q_i^*(\theta_i) \propto \exp\left(\mathbb{E}_{q_{-i}}[\log p(x, \theta)]\right)}$$

- **Mean-field**: $q = \prod_i q_i$ — factorization 가정
- **CAVI**: 좌표 업데이트로 ELBO 단조 증가
- **Conjugate-exponential**: 업데이트가 **닫힌형 exp family**
- **한계**: posterior correlation 무시 → variance underestimate

---

## 🤔 생각해볼 문제

**문제 1** (기초): 2D Gaussian $p(\theta_1, \theta_2) = \mathcal{N}(0, \begin{pmatrix}1 & \rho\\ \rho & 1\end{pmatrix})$에 mean-field $q = q_1(\theta_1)q_2(\theta_2)$의 최적 해는?

<details>
<summary>해설</summary>

Precision $\Sigma^{-1} = \frac{1}{1-\rho^2}\begin{pmatrix}1 & -\rho\\ -\rho & 1\end{pmatrix}$. Diagonal element $= 1/(1-\rho^2)$.

최적 $q_i = \mathcal{N}(0, (1-\rho^2))$.

참 marginal variance = 1. MF variance = $1 - \rho^2 < 1$.

**Underestimate**: correlation이 강할수록 차이 커짐(정리 3.4).

</details>

**문제 2** (심화): CAVI와 **Gibbs sampler**(Ch4-02)의 닮음과 차이는?

<details>
<summary>해설</summary>

| | CAVI | Gibbs |
|---|---|---|
| 업데이트 | $q_i^* \propto \exp\mathbb{E}_{q_{-i}}[\log p]$ | $\theta_i \sim p(\theta_i\|\theta_{-i}, x)$ |
| 본질 | 변분(deterministic) | 확률(sampling) |
| 수렴 | ELBO 단조 증가 | Posterior에 ergodic |
| 결과 | Factorized posterior 근사 | Exact posterior 샘플 |

**Gibbs = conditional에서 샘플, CAVI = conditional의 기댓값-기반 업데이트**. 수학적으로 매우 유사(같은 조건부 분포 이용), 철학적으로 다름.

Conjugate-exp 구조가 둘 다의 효율성의 원천.

</details>

**문제 3** (AI 연결): LDA에서 변분 CAVI가 **Gibbs sampling**보다 선호되는 경우는?

<details>
<summary>해설</summary>

- **대규모 데이터** (뉴스 아카이브 수백만 건): CAVI가 deterministic → 빠른 수렴, 병렬화 용이
- **Point estimate가 필요**: posterior mean만 있으면 됨 → CAVI
- **Real-time streaming**: online VI (Hoffman SVI, Ch2-04)

Gibbs 선호:
- **Posterior uncertainty 정확히**: multi-chain Gibbs + diagnostics
- **Tight coupling** (어려운 hierarchical): Collapsed Gibbs가 우수

실전 guidance: **빠른 prototype → CAVI, 최종 분석 → MCMC**로 이중 확인.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. ELBO의 3가지 분해](./02-elbo-three-decompositions.md) | [📚 README로 돌아가기](../README.md) | [04. Exponential Family와 자연매개변수 VI ▶](./04-exponential-family-vi.md) |

</div>
