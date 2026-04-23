# 03. Conjugate Priors의 수학

## 🎯 핵심 질문

- **Conjugate prior**란 무엇이고, 왜 존재 자체가 특별한가?
- Beta–Bernoulli, Gamma–Poisson, Normal–Normal 등 주요 쌍의 posterior는 어떻게 **닫힌형**으로 유도되는가?
- Conjugate 구조는 왜 **Exponential Family**에서 자연스럽게 나오는가?
- 실전에서 conjugate prior의 가치는? 비공액(non-conjugate) 모델이 왜 VI와 MCMC를 요구하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**CAVI**(Ch2-03)의 좌표 업데이트가 닫힌형이 되려면 **conjugate-exponential family** 관계가 필수. **Collapsed Gibbs**(Ch4-02)에서 latent를 주변화할 수 있는 것도 conjugate 관계 덕분. **Bayesian Linear Regression**, **LDA**(Latent Dirichlet Allocation), **Naive Bayes**의 posterior 모두 conjugate의 직접 결과. 그리고 VAE(Ch3-01)에서 Gaussian prior + Gaussian posterior KL이 **해석해**를 갖는 것도 Normal–Normal conjugate의 특수경우. 즉 "Conjugate 없이는 Bayesian ML의 수많은 공식이 존재하지 않았다".

---

## 📐 수학적 선행 조건

- [Ch1-01 Bayes 정리](./01-bayes-rule-four-roles.md), [Ch1-02 MLE/MAP/Bayesian](./02-mle-map-full-bayesian.md)
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Beta, Gamma, Dirichlet, Normal 분포
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Exponential Family, sufficient statistics
- 기초: Gamma 함수 $\Gamma(\cdot)$, Beta 함수 $B(\alpha, \beta) = \Gamma(\alpha)\Gamma(\beta)/\Gamma(\alpha+\beta)$

---

## 📖 직관적 이해

### Conjugate의 정의

Prior family $\mathcal{F}$이 likelihood에 대해 **conjugate**이면:

$$p(\theta) \in \mathcal{F} \implies p(\theta|D) \in \mathcal{F}$$

**같은 family 내에서 파라미터만 업데이트**되는 것. 즉 prior 모양을 그대로 유지하면서 데이터를 반영.

### 왜 특별한가

| 일반(비공액) | Conjugate |
|---|---|
| Posterior 형태 미지 | **Posterior가 같은 family 내** |
| Evidence 수치적분 | Evidence가 **정규화 상수의 비율** |
| MCMC/VI 필수 | **닫힌형 posterior** |
| Predictive도 수치적분 | Predictive도 **닫힌형** |
| 순차 업데이트 어려움 | **파라미터 업데이트**만 |

### 직관 — "Prior가 가상 데이터"

많은 conjugate 쌍에서 prior 파라미터는 **"가상의 이전 관측"**처럼 해석된다:
- Beta$(\alpha, \beta)$ = "이전에 $\alpha$번 성공, $\beta$번 실패 본 것처럼"
- Gamma$(a, b)$ for Poisson rate = "이전에 $a$번 사건을 $b$ 시간 관측한 것처럼"
- Normal prior mean = "이전에 관측한 $n_0$개 샘플의 평균"

이것이 **pseudo-count** 해석. 데이터가 무한히 쌓이면 prior의 영향이 사라지고 MLE로 수렴(BvM, Ch1-05).

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Conjugate Prior

확률분포 family $\mathcal{F}$이 likelihood $p(D|\theta)$에 대해 **conjugate**라는 것은:

$$p(\theta) \in \mathcal{F} \implies p(\theta|D) \in \mathcal{F} \quad \forall D$$

### 정의 3.2 — Exponential Family

분포 $p(x|\theta)$가 **exponential family**에 속한다는 것은:

$$p(x|\theta) = h(x)\exp\left(\eta(\theta)^T T(x) - A(\theta)\right)$$

- $\eta(\theta)$: **자연매개변수**(natural parameter)
- $T(x)$: **충분통계량**(sufficient statistic)
- $A(\theta)$: **log partition function** (정규화)
- $h(x)$: base measure

대부분의 "표준" 분포 — Bernoulli, Binomial, Poisson, Gaussian(두 파라미터), Gamma, Beta, Dirichlet — 모두 exponential family.

---

## 🔬 정리와 증명

### 정리 3.1 — Beta–Bernoulli Conjugate

**명제**: Bernoulli likelihood $p(x|\theta) = \theta^x(1-\theta)^{1-x}$와 Beta$(\alpha, \beta)$ prior에 대해, $D = (x_1, \ldots, x_n)$이고 $k = \sum x_i$일 때:

$$p(\theta|D) = \text{Beta}(\alpha + k, \beta + n - k)$$

**증명**:

$$p(D|\theta) = \prod_i \theta^{x_i}(1-\theta)^{1-x_i} = \theta^k(1-\theta)^{n-k}$$

$$p(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}$$

$$p(\theta|D) \propto p(D|\theta)p(\theta) \propto \theta^{\alpha+k-1}(1-\theta)^{\beta+n-k-1}$$

이것이 Beta$(\alpha + k, \beta + n - k)$의 unnormalized density. 정규화 상수는 $1/B(\alpha+k, \beta+n-k)$. $\square$

**해석**: Prior가 $(\alpha-1, \beta-1)$개의 성공/실패를 "본 것"처럼 → 데이터 $(k, n-k)$를 더해 업데이트.

### 정리 3.2 — Gamma–Poisson Conjugate

**명제**: Poisson likelihood $p(x|\lambda) = \lambda^x e^{-\lambda}/x!$와 Gamma$(a, b)$ prior (rate parametrization)에 대해, $D = (x_1, \ldots, x_n)$이면:

$$p(\lambda|D) = \text{Gamma}\left(a + \sum x_i,\ b + n\right)$$

**증명**:

$$p(D|\lambda) \propto \lambda^{\sum x_i}e^{-n\lambda}$$

$$p(\lambda) \propto \lambda^{a-1}e^{-b\lambda}$$

$$p(\lambda|D) \propto \lambda^{a + \sum x_i - 1}e^{-(b+n)\lambda} = \text{Gamma}(a + \sum x_i, b + n)$$

$\square$

### 정리 3.3 — Normal–Normal (Known Variance)

**명제**: $x_i|\mu \sim \mathcal{N}(\mu, \sigma^2)$ (알려진 $\sigma^2$), prior $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$이면:

$$p(\mu|D) = \mathcal{N}(\mu_n, \tau_n^2)$$

여기서

$$\tau_n^{-2} = \tau_0^{-2} + n\sigma^{-2}, \qquad \mu_n = \tau_n^2\left(\tau_0^{-2}\mu_0 + \sigma^{-2}\sum x_i\right)$$

**증명**:

Log posterior ($\mu$에 대한 2차식):

$$\log p(\mu|D) = -\frac{1}{2\tau_0^2}(\mu - \mu_0)^2 - \frac{1}{2\sigma^2}\sum(x_i - \mu)^2 + \text{const}$$

$\mu$에 대한 2차·1차 계수를 묶으면:

$$= -\frac{1}{2}\mu^2\left(\tau_0^{-2} + n\sigma^{-2}\right) + \mu\left(\tau_0^{-2}\mu_0 + \sigma^{-2}\sum x_i\right) + \text{const}$$

이것이 Gaussian의 완성형. precision $\tau_n^{-2} = \tau_0^{-2} + n\sigma^{-2}$, mean $\mu_n = \tau_n^2 \times (\cdot)$. $\square$

**Precision additivity**: "데이터가 많아질수록 posterior precision은 prior precision에 likelihood precision을 더해서 증가" — Fisher 정보의 additivity.

### 정리 3.4 — Normal–Inverse-Gamma (Unknown Mean & Variance)

**명제**: $x_i|\mu, \sigma^2 \sim \mathcal{N}(\mu, \sigma^2)$에 대해, $(\mu, \sigma^2)$의 conjugate prior는 **Normal-Inverse-Gamma**:

$$\sigma^2 \sim \text{Inv-Gamma}(a_0, b_0), \quad \mu|\sigma^2 \sim \mathcal{N}(\mu_0, \sigma^2/\kappa_0)$$

Posterior: $\sigma^2|D \sim \text{Inv-Gamma}(a_n, b_n)$, $\mu|\sigma^2, D \sim \mathcal{N}(\mu_n, \sigma^2/\kappa_n)$.

파라미터 업데이트:
- $\kappa_n = \kappa_0 + n$
- $\mu_n = (\kappa_0\mu_0 + n\bar x)/\kappa_n$
- $a_n = a_0 + n/2$
- $b_n = b_0 + \frac{1}{2}\sum(x_i - \bar x)^2 + \frac{\kappa_0 n(\bar x - \mu_0)^2}{2(\kappa_0 + n)}$

**증명 개요**: Log-joint를 $(\mu, \sigma^2)$에 대해 분해 → Inverse-Gamma × Gaussian 형태. 상세는 Gelman BDA3. $\square$

### 정리 3.5 — Dirichlet–Multinomial

**명제**: $x \sim \text{Mult}(n, \pi)$, $\pi \sim \text{Dir}(\alpha)$이면 $\pi|D \sim \text{Dir}(\alpha + x)$.

**증명**:

$$p(D|\pi) \propto \prod_k \pi_k^{x_k}, \quad p(\pi) \propto \prod_k \pi_k^{\alpha_k - 1}$$

$$p(\pi|D) \propto \prod_k \pi_k^{\alpha_k + x_k - 1} = \text{Dir}(\alpha + x) \quad \square$$

**응용**: LDA(Latent Dirichlet Allocation), n-gram 언어 모델의 smoothing.

### 정리 3.6 — Exponential Family Conjugate 일반론

**명제**: Likelihood가 exponential family $p(x|\theta) = h(x)\exp(\eta(\theta)^T T(x) - A(\theta))$이면, conjugate prior는:

$$p(\theta|\nu, \tau) \propto \exp(\nu^T \eta(\theta) - \tau A(\theta))$$

에서 뽑는다(특정 form). Posterior:

$$p(\theta|D, \nu, \tau) \propto \exp\left((\nu + \sum T(x_i))^T \eta(\theta) - (\tau + n) A(\theta)\right)$$

즉 **prior와 같은 exponential family, 파라미터만 $(\nu, \tau) \to (\nu + \sum T(x_i), \tau + n)$**.

**증명**:

$$p(D|\theta)p(\theta|\nu, \tau) \propto \left[\prod h(x_i)\right]\exp\left(\eta(\theta)^T\sum T(x_i) - nA(\theta)\right) \cdot \exp(\nu^T\eta(\theta) - \tau A(\theta))$$

$$\propto \exp\left((\nu + \sum T(x_i))^T \eta(\theta) - (\tau + n)A(\theta)\right)$$

$\square$

> **통합적 의미**: Beta, Gamma, Normal-NIG, Dirichlet 모두 이 일반 공식의 특수경우. **Sufficient statistic만 업데이트**하면 되므로 순차·온라인 학습 자연.

### 예시 — 주요 Conjugate 쌍 정리표

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Bernoulli($\theta$) | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Binomial($n, \theta$) | Beta($\alpha, \beta$) | Beta($\alpha + k, \beta + n - k$) |
| Poisson($\lambda$) | Gamma($a, b$) | Gamma($a + \sum x_i, b + n$) |
| Multinomial($\pi$) | Dirichlet($\alpha$) | Dirichlet($\alpha + x$) |
| Normal($\mu, \sigma^2$ known) | Normal($\mu_0, \tau_0^2$) | Normal($\mu_n, \tau_n^2$) |
| Normal($\mu$ known, $\sigma^2$) | Inv-Gamma($a, b$) | Inv-Gamma($a + n/2, b + \frac{1}{2}\sum(x_i-\mu)^2$) |
| Normal($\mu, \sigma^2$ both) | Normal-Inv-Gamma | Normal-Inv-Gamma (업데이트) |
| Normal multivariate | Normal-Inv-Wishart | Normal-Inv-Wishart |
| Exponential($\lambda$) | Gamma($a, b$) | Gamma($a + n, b + \sum x_i$) |
| Uniform(0, $\theta$) | Pareto($x_m, \alpha$) | Pareto($\max(x_m, \max x_i), \alpha + n$) |

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(42)

# ────────────────────────────────────────────────
# 1. Beta–Bernoulli
# ────────────────────────────────────────────────
alpha, beta_ = 2.0, 2.0
data = rng.binomial(1, 0.7, size=50)
k = data.sum()
post_a, post_b = alpha + k, beta_ + len(data) - k
print(f"Beta-Bernoulli: Beta({post_a:.0f}, {post_b:.0f})")

# ────────────────────────────────────────────────
# 2. Gamma–Poisson
# ────────────────────────────────────────────────
a0, b0 = 2.0, 1.0
counts = rng.poisson(5.0, size=30)
post_a_g, post_b_g = a0 + counts.sum(), b0 + len(counts)
print(f"Gamma-Poisson:  Gamma({post_a_g}, {post_b_g})  → mean = {post_a_g/post_b_g:.3f}")

# ────────────────────────────────────────────────
# 3. Normal–Normal (알려진 sigma)
# ────────────────────────────────────────────────
sigma = 1.0
mu0, tau0 = 0.0, 10.0  # prior
xs = rng.normal(2.0, sigma, size=20)

post_prec = 1/tau0**2 + len(xs)/sigma**2
post_var = 1/post_prec
post_mean = post_var * (mu0/tau0**2 + xs.sum()/sigma**2)
print(f"Normal-Normal:  N({post_mean:.3f}, {np.sqrt(post_var):.3f}²)")

# ────────────────────────────────────────────────
# 4. Dirichlet–Multinomial
# ────────────────────────────────────────────────
alpha_dir = np.array([1.0, 1.0, 1.0])
counts_m = np.array([10, 20, 5])
post_dir = alpha_dir + counts_m
print(f"Dir-Multinomial: Dir{tuple(post_dir)} → mean = {post_dir/post_dir.sum()}")

# ────────────────────────────────────────────────
# 5. 시각화 — Beta 업데이트 순차
# ────────────────────────────────────────────────
theta = np.linspace(0, 1, 500)
fig, ax = plt.subplots(figsize=(10, 5))
a_c, b_c = alpha, beta_
ax.plot(theta, stats.beta(a_c, b_c).pdf(theta), lw=2, label=f'Prior Beta({a_c:.0f},{b_c:.0f})')
for n_seen in [10, 25, 50]:
    k_s = data[:n_seen].sum()
    ax.plot(theta, stats.beta(alpha+k_s, beta_+n_seen-k_s).pdf(theta),
            lw=2, label=f'n={n_seen}, Beta({alpha+k_s:.0f},{beta_+n_seen-k_s:.0f})')
ax.axvline(0.7, color='k', ls='--', alpha=0.5, label='True θ=0.7')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel('density')
ax.set_title('Beta conjugate 업데이트 — 데이터가 쌓일수록 prior가 잊혀짐')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('conjugate_updates.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Bayesian Linear Regression
Gaussian noise + Gaussian prior on weights = Normal-Normal conjugate. Posterior는 $\mathcal{N}(\mu_n, \Sigma_n)$, $\mu_n = \Sigma_n(\sigma^{-2}X^T y + \Sigma_0^{-1}\mu_0)$ 닫힌형.

### LDA (Latent Dirichlet Allocation)
Dirichlet-Multinomial conjugate가 **core**. Topic-word 분포 $\phi_k$와 document-topic 분포 $\theta_d$가 모두 Dirichlet prior, Multinomial likelihood.

### Naive Bayes
Categorical features + Dirichlet prior = smoothing (Laplace smoothing = $\alpha = 1$ Dirichlet MAP).

### VAE의 KL 해석해
Gaussian prior $\mathcal{N}(0, I)$ + Gaussian posterior $\mathcal{N}(\mu, \sigma^2)$의 KL이 **closed form** — Normal-Normal conjugate 구조 덕분(Ch3-01).

### Online/Streaming 학습
Conjugate 구조 ⇒ 파라미터만 업데이트 ⇒ **진정한 온라인 학습**. Kalman filter = Normal-Normal의 순차적 적용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Conjugate family가 모델에 적절 | 실제 세계는 종종 **비공액** (NN likelihood 등) |
| Prior가 가상 데이터로 해석 가능 | 다차원·복잡 모델에서 prior 해석 어려움 |
| Prior의 informative 조절 가능 | 너무 diffuse하면 improper 또는 posterior가 이상 |
| 닫힌형 posterior | BNN 등 복잡 모델엔 적용 불가 → VI(Ch2), MCMC(Ch4) 필요 |
| Exponential family 적용 | Heavy-tail, mixture 등은 추가 장치 필요 |

**실전적 교훈**: Conjugate 관계는 **교육·prototype·baseline**에 최고. 복잡 모델은 VI/MCMC가 대신하지만, conjugate-exponential 구조가 있으면 CAVI·Collapsed Gibbs가 훨씬 효율적.

---

## 📌 핵심 정리

$$\boxed{\text{Conjugate: } p(\theta) \in \mathcal{F} \Rightarrow p(\theta|D) \in \mathcal{F}}$$

주요 쌍:
- **Beta–Bernoulli**: Beta$(\alpha+k, \beta+n-k)$
- **Gamma–Poisson**: Gamma$(a+\sum x, b+n)$
- **Normal–Normal**: precision 더하기 + precision-weighted mean
- **Dirichlet–Multinomial**: Dirichlet$(\alpha + x)$
- **Normal–Inv-Gamma**: 평균·분산 동시 추론

Exponential family에서 일반적으로: **sufficient statistic** $\sum T(x_i)$ 누적, **count** $\tau + n$.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Beta$(1,1)$ prior(=uniform)에서 Bernoulli 데이터 $k = 7, n = 10$의 posterior는? Posterior mode와 MLE를 비교하라.

<details>
<summary>해설</summary>

Posterior = Beta$(1+7, 1+3) = $Beta$(8, 4)$.

Mode = $(8-1)/(8+4-2) = 7/10 = 0.7$.

MLE = $k/n = 7/10 = 0.7$.

Uniform prior에서 **mode = MLE** (정리 2.1).

Mean = $8/12 \approx 0.667$ ≠ mode — Beta가 대칭이 아닐 때 둘이 다름.

</details>

**문제 2** (심화): Normal–Normal에서 "prior의 precision은 $n_0 = \sigma^2/\tau_0^2$개 가상 관측과 동치"라고 한다. 이 해석을 posterior mean 공식에서 직접 보여라.

<details>
<summary>해설</summary>

$\mu_n = \tau_n^2(\tau_0^{-2}\mu_0 + \sigma^{-2}\sum x_i)$

$n_0 := \sigma^2/\tau_0^2$라 놓으면 $\tau_0^{-2} = n_0/\sigma^2$:

$$\mu_n = \frac{\sigma^{-2}(n_0 \mu_0 + \sum x_i)}{\sigma^{-2}(n_0 + n)} = \frac{n_0 \mu_0 + \sum x_i}{n_0 + n} = \frac{n_0 \mu_0 + n\bar x}{n_0 + n}$$

이것이 "prior에서 $n_0$개 가상 관측(평균 $\mu_0$) + 실제 $n$개 관측(평균 $\bar x$)의 가중 평균" 형태. $n_0$가 prior strength.

</details>

**문제 3** (AI 연결): VAE에서 encoder가 $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$이고 prior가 $p(z) = \mathcal{N}(0, I)$이면 KL term이 $\frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$이 나온다. 이 공식과 Normal–Normal conjugate는 어떤 관계?

<details>
<summary>해설</summary>

VAE의 KL은 "$q$에서 $p$로의 divergence"이지 posterior 자체가 아님. 하지만 **두 Gaussian의 KL closed form**은 Normal-Normal conjugate 구조의 **산물**:

$$\text{KL}(\mathcal{N}(\mu_1, \Sigma_1)\|\mathcal{N}(\mu_2, \Sigma_2)) = \frac{1}{2}[\text{tr}(\Sigma_2^{-1}\Sigma_1) + (\mu_2 - \mu_1)^T\Sigma_2^{-1}(\mu_2 - \mu_1) - k + \log|\Sigma_2|/|\Sigma_1|]$$

$\mu_2 = 0, \Sigma_2 = I$이면 정확히 VAE 공식.

이 해석해가 없었다면 VAE는 **MC 추정으로 KL을 근사**해야 했을 것(분산 증가, 불안정). Gaussian conjugate 덕분에 **학습 안정·빠름**.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. MLE vs MAP vs Full Bayesian](./02-mle-map-full-bayesian.md) | [📚 README로 돌아가기](../README.md) | [04. Predictive Distribution ▶](./04-predictive-distribution.md) |

</div>
