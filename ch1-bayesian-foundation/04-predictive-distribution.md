# 04. Predictive Distribution

## 🎯 핵심 질문

- **Posterior predictive** $p(y^*|D)$와 **단일 점추정 predictive** $p(y^*|\hat\theta)$는 무엇이 다른가?
- 왜 예측 분산이 **epistemic**(모델) + **aleatoric**(노이즈)으로 분해되는가?
- Beta–Bernoulli에서 predictive가 왜 **Beta–Binomial**이 되는가?
- **Bayesian Model Averaging**(BMA)과 predictive의 관계는?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

"모델의 confidence"를 제대로 말하려면 **예측 분포의 분산**이 필요하다. 점추정은 "0.7 확률로 앞면"이라고 답하지만 이 0.7이 **얼마나 확실한가**는 모름. Bayesian predictive는 "0.7이 ±0.1 퍼져 있다"를 추가로 제공. 이것이 **OOD detection**, **active learning**, **Bayesian optimization**의 수학적 핵심. BNN(Ch5-01)의 예측이 불확실성을 주는 것도 정확히 predictive distribution의 계산.

---

## 📐 수학적 선행 조건

- [Ch1-01~03](./01-bayes-rule-four-roles.md): Posterior, conjugate priors
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Total variance decomposition, iterated expectation
- Beta-Binomial, Student's t, Negative Binomial 분포

---

## 📖 직관적 이해

### 점추정 vs Full predictive

| | Plug-in predictive $p(y^*\|\hat\theta)$ | Posterior predictive $p(y^*\|D)$ |
|---|---|---|
| 정의 | $p(y^*\|\hat\theta_{MLE/MAP})$ | $\int p(y^*\|\theta)p(\theta\|D)d\theta$ |
| 불확실성 | 관측 노이즈만 (aleatoric) | **+ 모델 불확실성** (epistemic) |
| 분포 분산 | 점추정의 variance | 일반적으로 **더 크다** |
| OOD에서 | 같음 (overconfident) | **넓어짐** — 정확한 behavior |

### 요리 비유

- **점추정**: "요리사가 김철수라 확신. 김철수가 오늘 김치찌개 낼 확률은 0.7."
- **Full predictive**: "요리사가 김철수(70%), 이영희(20%), 박민수(10%)일 것 같음. 각각의 메뉴 확률을 가중 평균."

### 분산 분해 (Law of Total Variance)

$$\text{Var}[y^*|D] = \underbrace{\mathbb{E}_{\theta|D}[\text{Var}(y^*|\theta)]}_{\text{aleatoric}} + \underbrace{\text{Var}_{\theta|D}[\mathbb{E}(y^*|\theta)]}_{\text{epistemic}}$$

- **Aleatoric**: 같은 $\theta$에서 관측 노이즈 — 데이터 많아도 감소 안 함
- **Epistemic**: $\theta$에 대한 불확실성 — **데이터 많아지면 감소**

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Posterior Predictive

$$p(y^*|D) = \int p(y^*|\theta)\,p(\theta|D)\,d\theta$$

새 관측 $y^*$가 데이터 $D$만 주어졌을 때의 marginal 분포. 모수 $\theta$를 **주변화**.

### 정의 4.2 — Prior Predictive

데이터를 보기 전:

$$p(y^*) = \int p(y^*|\theta)\,p(\theta)\,d\theta$$

BMA와 Bayesian model comparison(Ch1-01 정리 1.4)의 evidence $p(D)$가 prior predictive의 예.

### 정의 4.3 — Plug-in Predictive

점추정의 $p(y^*|\hat\theta)$:

$$p(y^*|D) \approx p(y^*|\hat\theta_{MLE \text{ or } MAP})$$

Posterior를 **delta 함수**로 근사한 특수경우(Ch1-02 정리 2.4).

---

## 🔬 정리와 증명

### 정리 4.1 — Total Variance Decomposition

**명제**:
$$\text{Var}[y^*|D] = \mathbb{E}_{\theta|D}[\text{Var}(y^*|\theta)] + \text{Var}_{\theta|D}[\mathbb{E}(y^*|\theta)]$$

**증명**:

정의로부터:
$$\text{Var}[y^*|D] = \mathbb{E}[(y^* - \mathbb{E}[y^*|D])^2 | D]$$

조건부 기댓값의 iterated law:
$$\mathbb{E}[(y^*)^2|D] = \mathbb{E}_\theta[\mathbb{E}[(y^*)^2|\theta, D] | D] = \mathbb{E}_\theta[\mathbb{E}[(y^*)^2|\theta]]$$

(관측 노이즈는 $\theta$만 조건하면 됨)

$$\mathbb{E}[(y^*)^2|\theta] = \text{Var}(y^*|\theta) + \mathbb{E}(y^*|\theta)^2$$

따라서:
$$\mathbb{E}[(y^*)^2|D] = \mathbb{E}_\theta[\text{Var}(y^*|\theta)] + \mathbb{E}_\theta[\mathbb{E}(y^*|\theta)^2]$$

한편:
$$\mathbb{E}[y^*|D]^2 = (\mathbb{E}_\theta[\mathbb{E}(y^*|\theta)])^2$$

빼면:
$$\text{Var}[y^*|D] = \mathbb{E}_\theta[\text{Var}(y^*|\theta)] + \underbrace{\mathbb{E}_\theta[\mathbb{E}(y^*|\theta)^2] - (\mathbb{E}_\theta[\mathbb{E}(y^*|\theta)])^2}_{\text{Var}_\theta[\mathbb{E}(y^*|\theta)]}$$

$\square$

> **Ch7-03에서 이 공식이 Epistemic/Aleatoric 분해의 공식적 정의가 된다.**

### 정리 4.2 — Beta–Bernoulli Predictive = Beta–Binomial

**명제**: Beta$(\alpha, \beta)$ prior, Bernoulli likelihood, $k$/$n$ 성공 관측 후, $m$번 추가로 던졌을 때 $y^*$번 성공할 확률:

$$p(y^*|D) = \binom{m}{y^*}\frac{B(\alpha+k+y^*, \beta+n-k+m-y^*)}{B(\alpha+k, \beta+n-k)}$$

이것은 **Beta–Binomial** 분포.

**증명**:

Posterior: $p(\theta|D) = \text{Beta}(\alpha+k, \beta+n-k)$ (Ch1-03 정리 3.1). 이를 $(a_n, b_n)$로 표기.

$$p(y^*|D) = \int_0^1 \binom{m}{y^*}\theta^{y^*}(1-\theta)^{m-y^*} \cdot \frac{\theta^{a_n-1}(1-\theta)^{b_n-1}}{B(a_n, b_n)}d\theta$$

$$= \binom{m}{y^*}\frac{1}{B(a_n, b_n)}\int_0^1 \theta^{a_n + y^* - 1}(1-\theta)^{b_n + m - y^* - 1}d\theta$$

$$= \binom{m}{y^*}\frac{B(a_n + y^*, b_n + m - y^*)}{B(a_n, b_n)}$$

$\square$

### 정리 4.3 — Normal–Normal Predictive (알려진 $\sigma^2$)

**명제**: Posterior $\mu|D \sim \mathcal{N}(\mu_n, \tau_n^2)$에서 새 관측 $y^* \sim \mathcal{N}(\mu, \sigma^2)$이면:

$$p(y^*|D) = \mathcal{N}(\mu_n, \tau_n^2 + \sigma^2)$$

**증명**:

$y^* = \mu + \epsilon$, $\mu \sim \mathcal{N}(\mu_n, \tau_n^2)$, $\epsilon \sim \mathcal{N}(0, \sigma^2)$ (독립).

두 Gaussian의 합이 Gaussian:
$$\mathbb{E}[y^*|D] = \mu_n, \quad \text{Var}[y^*|D] = \tau_n^2 + \sigma^2$$

$\square$

**해석**: 분산 $= \tau_n^2$ (epistemic: posterior 불확실성) $+ \sigma^2$ (aleatoric: 관측 노이즈) — 정리 4.1의 Gaussian 특수경우.

**Plug-in과 비교**: $p(y^*|\hat\mu) = \mathcal{N}(\hat\mu, \sigma^2)$ — $\tau_n^2$가 빠짐. 항상 **과소 추정**.

### 정리 4.4 — Normal–Inverse-Gamma Predictive = Student's t

**명제**: $\mu, \sigma^2$ 모두 미지, Normal-Inv-Gamma prior & posterior (정리 3.4)에서 predictive는 **Student's t 분포**:

$$p(y^*|D) = t_{2a_n}\left(\mu_n, \frac{b_n(\kappa_n + 1)}{a_n \kappa_n}\right)$$

자유도 $2a_n$, 평균 $\mu_n$, 스케일 $\sqrt{b_n(\kappa_n+1)/(a_n\kappa_n)}$.

**증명 개요**: Marginalize $\sigma^2$ out from Normal × Inv-Gamma → t 분포 정의. 상세는 BDA3 Appendix A. $\square$

**의미**: 미지 분산으로 인한 **heavy tail** — 소표본에서 Gaussian보다 tail이 두꺼움 (t의 자유도 효과).

### 정리 4.5 — Bayesian Model Averaging의 Predictive

**명제**: 여러 모델 $\{M_k\}$ 각각에 대한 posterior $p(\theta_k|D, M_k)$와 모델 사후 $p(M_k|D)$가 있을 때:

$$p(y^*|D) = \sum_k p(M_k|D) \int p(y^*|\theta_k, M_k)\,p(\theta_k|D, M_k)\,d\theta_k$$

즉 모델별 predictive의 **posterior-probability 가중합**.

**증명**: Iterated expectation $\mathbb{E}[y^*|D] = \mathbb{E}_{M|D}[\mathbb{E}[y^*|M, D]]$. $\square$

**핵심**: Best single model을 쓰지 않고 **모든 모델의 평균**을 쓰는 것이 예측 성능을 개선한다는 것이 BMA의 약속(Hoeting et al. 1999).

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Beta-Bernoulli predictive — Beta-Binomial
# ────────────────────────────────────────────────
alpha, beta_ = 2.0, 2.0
n, k = 10, 7
a_n, b_n = alpha + k, beta_ + n - k

# 새로 m=5번 던질 때, y*=0,1,...,5 각각의 확률
m = 5
y_star = np.arange(m + 1)
p_post_pred = np.array([
    stats.betabinom(m, a_n, b_n).pmf(y) for y in y_star
])

# Plug-in: theta_hat = 0.7 (MLE)
theta_hat = k / n
p_plugin = stats.binom(m, theta_hat).pmf(y_star)

print("y* | Posterior Predictive | Plug-in (MLE)")
for y in y_star:
    print(f"{y:>2} | {p_post_pred[y]:.4f}               | {p_plugin[y]:.4f}")

# 분산 비교
var_pp = (y_star**2 @ p_post_pred) - (y_star @ p_post_pred)**2
var_pi = (y_star**2 @ p_plugin) - (y_star @ p_plugin)**2
print(f"\nVariance — posterior predictive: {var_pp:.4f}")
print(f"Variance — plug-in:             {var_pi:.4f}")
print(f"(Epistemic uncertainty 추가로 인해 predictive 분산이 더 큼)")

# ────────────────────────────────────────────────
# Normal-Normal predictive — 분산 분해
# ────────────────────────────────────────────────
sigma, mu0, tau0 = 1.0, 0.0, 10.0
xs = rng.normal(2.0, sigma, size=8)
n_dat = len(xs)

post_prec = 1/tau0**2 + n_dat/sigma**2
tau_n2 = 1/post_prec
mu_n = tau_n2 * (mu0/tau0**2 + xs.sum()/sigma**2)

# Predictive variance = tau_n^2 (epistemic) + sigma^2 (aleatoric)
var_predictive = tau_n2 + sigma**2
print(f"\nμ_n = {mu_n:.3f}, τ_n² = {tau_n2:.4f}")
print(f"Predictive variance = {tau_n2:.4f} (epi) + {sigma**2:.4f} (ale) = {var_predictive:.4f}")

# ────────────────────────────────────────────────
# 시각화 — Posterior vs Predictive vs Plug-in
# ────────────────────────────────────────────────
y_grid = np.linspace(-4, 8, 400)
post_pred_density = stats.norm(mu_n, np.sqrt(var_predictive)).pdf(y_grid)
plugin_density = stats.norm(mu_n, sigma).pdf(y_grid)         # mean plug-in
post_density = stats.norm(mu_n, np.sqrt(tau_n2)).pdf(y_grid) # posterior of mu itself

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_grid, post_density, lw=2, label=f'Posterior p(μ|D) — τ_n={np.sqrt(tau_n2):.3f}')
ax.plot(y_grid, post_pred_density, lw=2, label=f'Predictive p(y*|D) — σ_pred={np.sqrt(var_predictive):.3f}')
ax.plot(y_grid, plugin_density, '--', lw=2, label=f'Plug-in p(y*|μ_n) — σ={sigma}')
ax.scatter(xs, np.zeros_like(xs), c='k', s=30, zorder=5, label='data')
ax.set_xlabel('value'); ax.set_ylabel('density')
ax.set_title('Posterior vs Posterior Predictive vs Plug-in Predictive')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('predictive.png', dpi=150); plt.show()
```

**해석**:
- Posterior predictive는 plug-in보다 **항상 넓다** (epistemic 추가).
- 데이터가 많아지면 $\tau_n^2 \to 0$ → predictive가 plug-in으로 수렴.

---

## 🔗 AI/ML 연결

### BNN의 예측 불확실성
$p(y^*|x^*, D) = \int p(y^*|x^*, W)p(W|D)dW$를 MC 샘플 $W^{(t)} \sim p(W|D)$로 근사 — MC Dropout의 $\frac{1}{T}\sum p(y^*|x^*, W^{(t)})$가 이것(Ch5-04).

### Active Learning
Predictive variance가 큰 $x^*$를 우선 labeling → BNN + acquisition function = BALD (Bayesian Active Learning by Disagreement).

### Bayesian Optimization
GP의 predictive $\mathcal{N}(\mu_n(x), \sigma_n^2(x))$를 이용한 acquisition (EI, UCB) — Ch6 핵심.

### OOD Detection
OOD 점에서 **epistemic uncertainty가 커져서** predictive 분포가 넓어짐. In-distribution에서는 좁음. 이 차이가 OOD 분리의 신호(Ch7-04).

### Ensembles
Deep ensembles는 여러 NN의 예측을 평균 — 이산 BMA의 한 형태. "Bayesian approximation" 해석.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Posterior 적분 가능 | BNN처럼 고차원이면 MC 근사 필요 |
| Model $\mathcal{M}$이 옳음 | Model misspecification → 틀린 predictive |
| Aleatoric $\sigma^2$가 상수 | Heteroscedastic 시 $\sigma^2(x)$ 별도 추정 |
| IID 관측 | Time-series·correlated 시 수정 필요 |
| BMA가 더 낫다 | 실전에선 best-single-model 성능도 경쟁적 |

**실무 팁**: "점추정으로 충분한가?" = aleatoric만 신경 쓸 때 OK. "OOD/소표본/의사결정 중요?" = full predictive 필수.

---

## 📌 핵심 정리

$$\boxed{p(y^*|D) = \int p(y^*|\theta)p(\theta|D)d\theta}$$

$$\boxed{\text{Var}[y^*|D] = \underbrace{\mathbb{E}_\theta[\text{Var}(y^*|\theta)]}_{\text{aleatoric}} + \underbrace{\text{Var}_\theta[\mathbb{E}(y^*|\theta)]}_{\text{epistemic}}}$$

주요 닫힌형:
- **Beta-Bernoulli → Beta-Binomial**
- **Normal-Normal (known σ) → Normal(μ_n, τ_n² + σ²)**
- **Normal-Inv-Gamma → Student's t**
- **Gamma-Poisson → Negative Binomial**

---

## 🤔 생각해볼 문제

**문제 1** (기초): Beta$(1,1)$ prior, 동전 1번 던져 앞면($k=1, n=1$). 추가 1번의 predictive는?

<details>
<summary>해설</summary>

Posterior = Beta$(2, 1)$. Predictive for $m=1$:

$$p(y^*=1|D) = \int_0^1 \theta \cdot \text{Beta}(2,1)(\theta)d\theta = \frac{2}{3}$$

**Plug-in** (MLE $\hat\theta = 1$): $p(y^*=1) = 1$ ← 극단적 overconfidence.

이것이 Bayesian predictive의 이점 — **한 번 던져 앞면이라고 "100% 앞면"이라 말하지 않음**. Laplace의 rule of succession.

</details>

**문제 2** (심화): Predictive 분산에서 aleatoric 부분 $\mathbb{E}_\theta[\text{Var}(y^*|\theta)]$이 데이터 많아져도 줄지 않는 이유를 직관적으로.

<details>
<summary>해설</summary>

Aleatoric = 같은 $\theta$ 하에서도 $y^*$의 고유 무작위성(관측 노이즈). 모델을 완벽히 알아도 동전 던지기가 매번 다르듯.

수학적으로 $n \to \infty$에서 posterior가 delta로 수렴하면:
$$\mathbb{E}_\theta[\text{Var}(y^*|\theta)] \to \text{Var}(y^*|\theta_0) > 0$$

반면 epistemic $\text{Var}_\theta[\mathbb{E}(y^*|\theta)]$는 **posterior가 좁아지면서 0으로**. 이것이 "epistemic은 데이터로 줄고, aleatoric은 안 준다"의 수학적 근거. Ch7-03에서 이 구분이 BNN practice의 핵심.

</details>

**문제 3** (AI 연결): MC Dropout(Ch5-04)에서 $p(y^*|x^*, D) \approx \frac{1}{T}\sum_t p(y^*|x^*, W^{(t)})$를 쓴다. 이 추정기는 정리 4.5의 BMA와 어떻게 대응하는가?

<details>
<summary>해설</summary>

MC Dropout: 서로 다른 dropout mask → 서로 다른 "sub-network" $W^{(t)}$. 이를 모델 $M_t$로 보면:

$$\frac{1}{T}\sum_t p(y^*|x^*, W^{(t)}) \approx \int p(y^*|x^*, W) p(W|D) dW$$

Monte Carlo 추정. 정리 4.5의 연속 적분 BMA를 **이산 샘플로 근사**한 것. 

$p(M_k|D)$는 uniform (각 dropout 실현이 같은 weight) — "uniform-weight BMA". 딥러닝에서의 "ensemble" 접근의 Bayesian 해석.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Conjugate Priors의 수학](./03-conjugate-priors.md) | [📚 README로 돌아가기](../README.md) | [05. Posterior의 점근 성질 (Bernstein–von Mises) ▶](./05-bernstein-von-mises.md) |

</div>
