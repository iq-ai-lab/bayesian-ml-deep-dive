# 01. VI의 아이디어와 ELBO 유도

## 🎯 핵심 질문

- Intractable posterior $p(\theta|x)$를 어떻게 tractable family $q_\phi$로 근사하는가?
- $\min_q \text{KL}(q\|p(\cdot|x))$와 $\max_q \text{ELBO}$가 왜 **동치**인가?
- Jensen 부등식으로부터 ELBO를 어떻게 유도하는가?
- VI의 "**optimization-as-inference**" 철학이란?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**VAE**(Ch3-01)의 훈련 목적함수가 바로 ELBO — 없으면 $\log p(x)$ 직접 최적화 불가. **BNN**의 variational posterior(Ch5-03)도 ELBO. **Diffusion Model**의 ELBO는 DDPM 손실의 이론적 근거(Ch7-01). **Stan/PyMC**의 ADVI도 ELBO 기반. 즉 **현대 Bayesian DL의 거의 모든 scalable method는 ELBO에 의존**.

---

## 📐 수학적 선행 조건

- [Ch1-01 Bayes 정리](../ch1-bayesian-foundation/01-bayes-rule-four-roles.md): evidence $p(x)$의 intractability
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence, entropy
- Jensen 부등식: $\mathbb{E}[\phi(X)] \geq \phi(\mathbb{E}[X])$ for convex $\phi$
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): gradient ascent

---

## 📖 직관적 이해

### VI의 기본 아이디어

MCMC는 **샘플링**으로 posterior에 접근 — 정확하지만 느림. VI는 **최적화**로 근사 — 빠르지만 근사:

$$\underbrace{p(\theta|x)}_{\text{intractable}} \approx \underbrace{q_\phi(\theta)}_{\text{tractable family}}$$

"**posterior에 가장 가까운** $q_\phi$를 찾자". "가까움"의 척도는 **KL divergence**.

### 두 KL 방향

| KL 방향 | 이름 | 특성 |
|---------|------|------|
| $\text{KL}(q\|p)$ | **reverse KL** (VI 표준) | Mode-seeking, posterior의 한 mode에 집중 |
| $\text{KL}(p\|q)$ | **forward KL** (EP) | Mode-covering, posterior 전체를 덮음 |

VI는 거의 **reverse KL**만 쓴다 — $p(\theta|x)$를 모르기 때문 ($\text{KL}(p\|q)$는 $p$ 샘플 필요).

### ELBO = 증거 하한

$$\log p(x) = \mathcal{L}(q) + \text{KL}(q(\theta)\|p(\theta|x))$$

- 좌변 $\log p(x)$: evidence (상수, but intractable)
- KL $\geq 0$, 등식 when $q = p(\cdot|x)$

따라서 **ELBO $\mathcal{L}(q) \leq \log p(x)$** (lower bound on log evidence).

**KL을 최소화하는 것과 ELBO를 최대화하는 것이 동치** — $\log p(x)$가 상수이므로.

### 요리 비유

- 정확한 posterior: 모든 요리사에 대한 "정확한" 확률 (계산 불가)
- $q_\phi$: "요리사가 Gaussian 분포에 따라 온다"고 가정 + 평균·분산을 조절
- ELBO: "이 Gaussian이 실제 분포를 얼마나 잘 맞추는가"의 점수

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Variational Family

**Variational family** $\mathcal{Q} = \{q_\phi : \phi \in \Phi\}$는 tractable 분포들의 parametric family. 예:
- Mean-field Gaussian: $q_\phi(\theta) = \prod_i \mathcal{N}(\theta_i; \mu_i, \sigma_i^2)$, $\phi = \{\mu_i, \sigma_i\}$
- Full-rank Gaussian: $q_\phi(\theta) = \mathcal{N}(\mu, \Sigma)$
- Normalizing flow(Ch3-03): 더 유연한 family

### 정의 1.2 — KL Divergence

$$\text{KL}(q\|p) = \int q(\theta)\log\frac{q(\theta)}{p(\theta)}d\theta = \mathbb{E}_q[\log q - \log p]$$

성질: $\text{KL} \geq 0$ (Gibbs), $\text{KL}(q\|p) = 0 \iff q = p$ a.e.

### 정의 1.3 — Evidence Lower Bound (ELBO)

$$\mathcal{L}(q) = \mathbb{E}_{q(\theta)}[\log p(x, \theta) - \log q(\theta)]$$

동치 표현:
$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x|\theta)] - \text{KL}(q(\theta)\|p(\theta))$$

$$\mathcal{L}(q) = \log p(x) - \text{KL}(q(\theta)\|p(\theta|x))$$

(Ch2-02에서 세 분해를 자세히 유도)

### 정의 1.4 — VI 문제

$$q^* = \arg\max_{q \in \mathcal{Q}} \mathcal{L}(q) = \arg\min_{q \in \mathcal{Q}} \text{KL}(q(\theta)\|p(\theta|x))$$

---

## 🔬 정리와 증명

### 정리 1.1 — ELBO와 KL의 관계

**명제**:
$$\log p(x) = \mathcal{L}(q) + \text{KL}(q(\theta)\|p(\theta|x))$$

**증명**:

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] - \mathbb{E}_q[\log q(\theta)]$$

$p(x, \theta) = p(\theta|x)p(x)$로 분해:
$$= \mathbb{E}_q[\log p(\theta|x) + \log p(x)] - \mathbb{E}_q[\log q(\theta)]$$

$\log p(x)$는 $\theta$에 무관:
$$= \log p(x) + \mathbb{E}_q[\log p(\theta|x) - \log q(\theta)]$$

$$= \log p(x) - \mathbb{E}_q[\log q(\theta) - \log p(\theta|x)]$$

$$= \log p(x) - \text{KL}(q\|p(\cdot|x))$$

$\square$

### 정리 1.2 — ELBO는 Lower Bound

**명제**: $\mathcal{L}(q) \leq \log p(x)$, 등호 iff $q(\theta) = p(\theta|x)$ a.e.

**증명**: 정리 1.1에서 $\text{KL} \geq 0$, 등호 iff 분포가 같음. $\square$

### 정리 1.3 — Jensen 부등식으로부터의 ELBO 유도

**명제**: Jensen 부등식 $\log \mathbb{E}[X] \geq \mathbb{E}[\log X]$ (concave $\log$)에서 ELBO가 직접 유도됨:

$$\log p(x) = \log \int p(x, \theta)d\theta = \log \int q(\theta)\frac{p(x, \theta)}{q(\theta)}d\theta$$

$$= \log \mathbb{E}_q\left[\frac{p(x, \theta)}{q(\theta)}\right] \geq \mathbb{E}_q\left[\log \frac{p(x, \theta)}{q(\theta)}\right] = \mathcal{L}(q)$$

**증명**: 위 유도가 직접 증명. Jensen 부등식 등호 조건은 $p(x,\theta)/q(\theta) = \text{const}$, 즉 $q(\theta) \propto p(x, \theta) = p(\theta|x) p(x)$, 즉 **$q = p(\cdot|x)$**. $\square$

### 정리 1.4 — ELBO 최대화 = KL 최소화

**명제**: $\log p(x)$가 $\phi$에 무관한 상수이므로:

$$\arg\max_\phi \mathcal{L}(q_\phi) = \arg\min_\phi \text{KL}(q_\phi(\theta)\|p(\theta|x))$$

**증명**: 정리 1.1로 직접. $\square$

### 정리 1.5 — Optimality Gap

**명제**: VI의 approximation error:

$$\text{KL}(q^*\|p(\cdot|x)) = \log p(x) - \mathcal{L}(q^*)$$

즉 **ELBO와 $\log p(x)$의 차이**가 approximation quality의 지표.

**증명**: 정리 1.1에서 직접. $\square$

### 예시 — 1D Gaussian posterior

Posterior $p(\theta|x) = \mathcal{N}(\mu_p, \sigma_p^2)$, variational $q_\phi(\theta) = \mathcal{N}(\mu_q, \sigma_q^2)$.

$$\text{KL}(q\|p) = \log \frac{\sigma_p}{\sigma_q} + \frac{\sigma_q^2 + (\mu_q - \mu_p)^2}{2\sigma_p^2} - \frac{1}{2}$$

Minimize → $\mu_q = \mu_p, \sigma_q = \sigma_p$. **Family 안에 $p$가 있으면 정확한 회복**.

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Beta posterior를 Gaussian (in logit space)으로 VI
# ────────────────────────────────────────────────
# Data: 동전 10번 중 7번 앞면, Beta(2,2) prior → posterior Beta(9,5)
alpha, beta_ = 2.0, 2.0
n, k = 10, 7
a_n, b_n = alpha + k, beta_ + n - k

# True posterior
theta_grid = np.linspace(1e-3, 1-1e-3, 400)
post_true = stats.beta(a_n, b_n).pdf(theta_grid)

# VI objective (in logit space)
def neg_elbo(params, n_mc=2000, seed=1):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    rng_local = np.random.default_rng(seed)
    eps = rng_local.standard_normal(n_mc)
    u = mu + sigma * eps
    theta = 1.0 / (1.0 + np.exp(-u))
    # log p(x|theta) + log p(theta) — unnormalized posterior
    log_lik = k*np.log(theta) + (n-k)*np.log(1-theta)
    log_prior = (alpha-1)*np.log(theta) + (beta_-1)*np.log(1-theta)
    # logit Jacobian: dθ/du = θ(1-θ)
    log_jac = np.log(theta) + np.log(1 - theta)
    log_joint = log_lik + log_prior + log_jac  # log p(x, u)
    log_q = -0.5*eps**2 - log_sigma - 0.5*np.log(2*np.pi)
    elbo_mc = (log_joint - log_q).mean()
    return -elbo_mc

result = minimize(neg_elbo, [0.0, 0.0], method='Nelder-Mead',
                  options={'xatol': 1e-5, 'fatol': 1e-5})
mu_vi, log_sigma_vi = result.x
print(f"VI params: μ={mu_vi:.4f}, σ={np.exp(log_sigma_vi):.4f}")

# ELBO 값
log_evidence = np.log(stats.beta(alpha, beta_).pdf(1).sum()) # not directly usable
# 실제 log p(x) = log Beta-Binomial normalizing
from scipy.special import betaln
log_px = betaln(a_n, b_n) - betaln(alpha, beta_)
elbo = -result.fun
print(f"ELBO        : {elbo:.4f}")
print(f"log p(x)    : {log_px:.4f}")
print(f"Gap (= KL)  : {log_px - elbo:.4f}")

# Transform back to θ space
eps = rng.standard_normal(50000)
u = mu_vi + np.exp(log_sigma_vi)*eps
theta_vi_samples = 1.0 / (1.0 + np.exp(-u))

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta_grid, post_true, 'k-', lw=2.5, label='True posterior Beta(9,5)')
ax.hist(theta_vi_samples, bins=60, density=True, alpha=0.4, label='VI approximation')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel('density')
ax.set_title(f'VI — ELBO={elbo:.3f}, log p(x)={log_px:.3f}, KL gap={log_px-elbo:.3f}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('vi_elbo.png', dpi=150); plt.show()
```

**관찰**: VI가 posterior를 잘 근사하지만 KL gap이 존재 — logit-Gaussian family의 한계.

---

## 🔗 AI/ML 연결

### VAE
$q_\phi(z|x)$ = encoder network, ELBO = VAE loss. Reconstruction + KL regularization (Ch3-01).

### Bayesian NN
$q_\phi(W)$ = factorized Gaussian over weights, ELBO gradient로 학습 (Ch5-03 Bayes by Backprop).

### ADVI (Automatic Differentiation VI)
Stan·PyMC의 자동 VI. 사용자가 모델만 지정하면 내부적으로 mean-field Gaussian + reparameterization(Ch2-05) + SGD.

### MC Dropout = Approximate VI
Dropout의 Bernoulli noise가 variational posterior, test-time dropout이 ELBO 최대화의 암묵적 결과(Ch5-04).

### Diffusion Model
DDPM의 손실이 hierarchical ELBO의 재표현 — VAE가 1단계, Diffusion은 T단계(Ch7-01).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Variational family $\mathcal{Q}$ 표현력 | Mean-field는 correlation 없음 → underestimate variance |
| ELBO 최적화 수렴 | 비볼록 → local optima |
| Reverse KL (mode-seeking) | Multimodal posterior에서 한 mode만 |
| KL 계산 tractable | 복잡 prior는 MC 추정 필요 (증가된 분산) |
| $q$ samplable | Discrete latent → REINFORCE (Ch2-06) |

**실전 교훈**: VI는 빠르지만 **posterior의 핵심 특성**(covariance·multimodality)을 놓치기 쉬움. 정확도 중요시 MCMC(Ch4).

---

## 📌 핵심 정리

$$\boxed{\log p(x) = \mathcal{L}(q) + \text{KL}(q\|p(\cdot|x))}$$

$$\boxed{\max_q \mathcal{L}(q) \Leftrightarrow \min_q \text{KL}(q\|p(\cdot|x))}$$

핵심:
- **Evidence** = **ELBO** + **KL gap**
- $\mathcal{L}(q) = \log \mathbb{E}_q[p(x,\theta)/q(\theta)] \geq \mathbb{E}_q[\log p(x,\theta)/q(\theta)]$ (Jensen)
- **Optimization-as-inference**: posterior inference를 optimization으로 환원
- Variational family 선택이 approximation quality를 결정

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\mathcal{Q}$가 $p(\theta|x)$를 포함하면 VI의 정확한 해는?

<details>
<summary>해설</summary>

$q^* = p(\theta|x)$, KL gap = 0, ELBO = $\log p(x)$. **정확한 inference** 회복.

Conjugate 모델에서 $q$를 같은 family로 놓으면 일어남. 예: Normal-Normal에서 $q = \mathcal{N}(\mu_q, \tau_q^2)$, 최적 $(\mu_q, \tau_q^2) = (\mu_n, \tau_n^2)$.

</details>

**문제 2** (심화): Jensen 부등식 증명에서 등호 조건이 **$p(x,\theta)/q(\theta)$ = const**라 했다. 이 조건이 왜 $q = p(\cdot|x)$와 동치인가?

<details>
<summary>해설</summary>

$p(x,\theta)/q(\theta) = c$ 이면 $q(\theta) = p(x,\theta)/c$. $q$가 정규화된 확률분포이려면 $\int q = 1 \Rightarrow c = p(x)$.

즉 $q(\theta) = p(x,\theta)/p(x) = p(\theta|x)$. ✓

다른 말로: Jensen은 log의 concavity에서 오는 부등식 — 피적분 함수가 **상수**일 때만 등호. "posterior와 정확히 같을 때만"의 수학적 정식화.

</details>

**문제 3** (AI 연결): VAE의 encoder $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$는 "**amortized inference**"(Ch3-04)라 불린다. 각 $x_i$마다 별도 최적화 안 하고 NN 하나로 inference를 공유. Variational family 관점에서 이게 왜 제약인가?

<details>
<summary>해설</summary>

Non-amortized VI: 각 $x_i$에 대해 $q_i = \mathcal{N}(\mu_i, \sigma_i^2)$를 개별 최적화 → 더 정확.

Amortized VI: NN $q_\phi(\cdot|x)$가 **모든 $x$에 공통 함수**. 학습 가능 parameter는 NN weight $\phi$로 **유한**.

제약:
- 한 $x$의 sub-optimal $q$ 선택이 다른 $x$에 영향
- NN의 표현력이 한정되면 "**amortization gap**" 생김

장점: **inference 시 단일 forward pass** — 대규모에서 필수. Ch3-04에서 이 trade-off를 자세히.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch1-05 Bernstein–von Mises](../ch1-bayesian-foundation/05-bernstein-von-mises.md) | [📚 README로 돌아가기](../README.md) | [02. ELBO의 3가지 분해 ▶](./02-elbo-three-decompositions.md) |

</div>
