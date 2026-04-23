# 02. MLE vs MAP vs Full Bayesian

## 🎯 핵심 질문

- MLE, MAP, Full Bayesian 세 접근은 어떻게 **점진적으로 일반화**되는가?
- 왜 **MLE는 uniform prior의 MAP**이고, **MAP은 delta prior의 full Bayesian**인가?
- L2 regularization이 왜 **Gaussian prior의 MAP**과 정확히 같은가?
- 각 접근의 **불확실성 정량화 범위**는 어떻게 다르고, 언제 어느 것을 써야 하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

딥러닝 practitioner가 매일 쓰는 weight decay, Bayesian이 쓰는 Gaussian prior, 통계학자의 ridge regression — **이 세 가지가 수학적으로 같다**. 이 등식을 모르면 "regularization을 더 강하게"가 "prior를 더 좁게"와 같은 뜻임을 놓친다. BNN에서 "왜 Gaussian prior를 기본으로 쓰는가"의 답도 여기서 나온다(Ch5-02). **Dropout = Bernoulli prior MAP**, **L1 = Laplace prior MAP** 같은 정체성도 모두 이 프레임으로 통합된다.

---

## 📐 수학적 선행 조건

- [Ch1-01 Bayes 정리와 4가지 역할](./01-bayes-rule-four-roles.md): posterior ∝ likelihood × prior
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): MLE 정의·Fisher 정보
- [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive): gradient descent, Lagrange
- [Regularization Theory Deep Dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive): L2/L1 reg의 최적화 관점

---

## 📖 직관적 이해

### 세 접근의 "확신의 정도" 스펙트럼

| 접근 | 출력 | 불확실성 정량화 | 정보량 요구 |
|------|------|----------------|-------------|
| **MLE** | 점추정 $\hat\theta_{MLE}$ | ❌ (표본분포는 별도 유도) | likelihood만 |
| **MAP** | 점추정 $\hat\theta_{MAP}$ | ❌ | likelihood + prior |
| **Full Bayesian** | 분포 $p(\theta\|D)$ | ✅ | likelihood + prior |

MLE는 "데이터가 가장 잘 설명하는 $\theta$"만, MAP은 "prior + data가 가장 선호하는 $\theta$"만, Full Bayesian은 "$\theta$에 대한 모든 불확실성 분포"를 반환한다.

### 통합적 관점 — Prior의 "정보량"

$$\text{MLE} \xleftarrow{\text{uniform prior}} \text{MAP} \xleftarrow{\text{delta prior of full Bayes}} \text{Full Bayesian}$$

- **MLE**: prior가 "아무 정보도 없음"(uniform) → MAP 특수경우
- **MAP**: full Bayesian의 mode만 취함 → 분포를 **delta**로 축소
- **Full Bayesian**: prior 완전 사용 + posterior 분포 전체 유지

### 요리 비유

- **MLE**: "재료 $D$가 주어졌을 때, 어느 요리사가 가장 이 재료를 썼을까?"의 한 명만 지목
- **MAP**: "이 집에서 자주 오는 요리사의 경향(prior) + 재료(data)를 종합해서 가장 유력한 한 명"
- **Full Bayesian**: "모든 요리사에 대한 확률분포 전체" — 이 분포로 신뢰구간·예측·의사결정 가능

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Maximum Likelihood Estimator (MLE)

데이터 $D$와 likelihood $p(D|\theta)$에 대해:

$$\hat\theta_{MLE} = \arg\max_{\theta \in \Theta} p(D|\theta) = \arg\max_{\theta} \log p(D|\theta)$$

(단조 변환 $\log$는 argmax를 바꾸지 않는다.)

### 정의 2.2 — Maximum A Posteriori (MAP)

Prior $p(\theta)$를 도입하면:

$$\hat\theta_{MAP} = \arg\max_{\theta \in \Theta} p(\theta|D) = \arg\max_{\theta}\left[\log p(D|\theta) + \log p(\theta)\right]$$

Evidence $p(D)$가 $\theta$에 무관하므로 제외.

### 정의 2.3 — Full Bayesian

Posterior 분포 자체를 다룸:

$$p(\theta|D) = \frac{p(D|\theta)\,p(\theta)}{p(D)}$$

이로부터 다음 통계량을 계산:
- **Posterior mean**: $\mathbb{E}[\theta|D] = \int \theta\,p(\theta|D)\,d\theta$
- **Posterior variance**: $\text{Var}[\theta|D]$
- **Credible interval**: $[a, b]$ with $P(\theta \in [a,b]|D) = 1-\alpha$
- **Predictive**: $p(y^*|D) = \int p(y^*|\theta)p(\theta|D)d\theta$

---

## 🔬 정리와 증명

### 정리 2.1 — MLE = Uniform Prior의 MAP

**명제**: $\Theta$가 유계이고 $p(\theta) = 1/|\Theta|$ (uniform)이면:

$$\hat\theta_{MAP} = \hat\theta_{MLE}$$

**증명**:

$$\hat\theta_{MAP} = \arg\max_\theta[\log p(D|\theta) + \log p(\theta)]$$

$p(\theta) = c$ (상수)이면 $\log p(\theta) = \log c$도 상수. argmax에 영향 없음:

$$= \arg\max_\theta \log p(D|\theta) = \hat\theta_{MLE} \quad \square$$

**비유계 $\Theta$의 경우**: improper prior $p(\theta) \propto 1$로 형식적으로 확장 가능하지만, posterior가 proper인지 별도 확인 필요.

### 정리 2.2 — L2 Regularization = Gaussian Prior MAP

**명제**: 회귀 모형 $y_i = f(x_i; \theta) + \epsilon_i$, $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$과 prior $\theta \sim \mathcal{N}(0, \tau^2 I)$을 가정하자. 그러면 MAP 추정은:

$$\hat\theta_{MAP} = \arg\min_\theta \left[\sum_i (y_i - f(x_i;\theta))^2 + \frac{\sigma^2}{\tau^2}\|\theta\|_2^2\right]$$

즉 **L2-regularized least squares**이고, **$\lambda = \sigma^2/\tau^2$**가 regularization 강도.

**증명**:

Likelihood:
$$p(D|\theta) = \prod_i \frac{1}{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i - f(x_i;\theta))^2}{2\sigma^2}\right)$$

$$\log p(D|\theta) = -\frac{1}{2\sigma^2}\sum_i (y_i - f(x_i;\theta))^2 + \text{const}$$

Prior:
$$\log p(\theta) = -\frac{1}{2\tau^2}\|\theta\|_2^2 + \text{const}$$

MAP objective:
$$\log p(\theta|D) \propto \log p(D|\theta) + \log p(\theta)$$

$$= -\frac{1}{2\sigma^2}\sum_i (y_i - f(x_i;\theta))^2 - \frac{1}{2\tau^2}\|\theta\|_2^2 + \text{const}$$

음수를 붙이고 $2\sigma^2$로 스케일하면:

$$\hat\theta_{MAP} = \arg\min_\theta \left[\sum_i (y_i - f(x_i;\theta))^2 + \frac{\sigma^2}{\tau^2}\|\theta\|_2^2\right]$$

$\square$

> **중요 귀결**:
> - **Weight decay** in NN = Gaussian prior on weights
> - **Ridge regression** = MAP with Gaussian prior
> - **Early stopping** ≈ implicit Gaussian prior (argued in Neal 1996)

### 정리 2.3 — L1 Regularization = Laplace Prior MAP

**명제**: Prior $p(\theta) = \prod_j \frac{1}{2b}\exp(-|\theta_j|/b)$ (Laplace)이면 MAP은:

$$\hat\theta_{MAP} = \arg\min_\theta\left[\sum_i(y_i - f_i(\theta))^2 + \frac{\sigma^2}{b}\|\theta\|_1\right]$$

즉 **LASSO**.

**증명**: 정리 2.2와 동일한 방식. Laplace prior의 log-density가 $-\|\theta\|_1/b$에 비례. $\square$

### 정리 2.4 — MAP은 Delta-posterior의 점추정

**명제**: Full Bayesian posterior를 **delta 함수** $\delta(\theta - \hat\theta_{MAP})$로 근사하면, predictive $p(y^*|D) \approx p(y^*|\hat\theta_{MAP})$가 된다. 즉 **MAP = 불확실성을 완전히 버린 Bayesian**.

**증명**:
$$p(y^*|D) = \int p(y^*|\theta)\,p(\theta|D)\,d\theta \approx \int p(y^*|\theta)\,\delta(\theta - \hat\theta_{MAP})\,d\theta = p(y^*|\hat\theta_{MAP})$$

$\square$

**귀결**: MAP으로 예측하면 **epistemic uncertainty**(Ch7-03)를 전부 잃어버림. 이것이 BNN이 MAP 대신 posterior 전체를 쓰는 이유.

### 정리 2.5 — Reparameterization 비불변성 (Invariance Failure of MAP)

**명제**: MLE는 **reparameterization 불변**(one-to-one $\phi = g(\theta)$에서 $\hat\phi_{MLE} = g(\hat\theta_{MLE})$)이지만, **MAP은 그렇지 않다**.

**증명**:

변수변환 공식: $p_\phi(\phi) = p_\theta(g^{-1}(\phi))|J|$ where $J$ is Jacobian.

Posterior도 Jacobian이 따라붙으므로:
$$\hat\phi_{MAP} = \arg\max p(\phi|D) = \arg\max[p(D|\phi) p_\phi(\phi)]$$
$$= \arg\max[p(D|g^{-1}(\phi)) p_\theta(g^{-1}(\phi)) |J|]$$

일반적으로 $g(\hat\theta_{MAP}) \neq \hat\phi_{MAP}$. $\square$

> **Full Bayesian은 이 문제 없음**: posterior 분포 자체가 변수변환에 맞춰 움직임.

### 예시

**예시 1 — Beta-Bernoulli**:
- Prior Beta$(\alpha, \beta)$, likelihood Binomial$(n, \theta)$의 $k$번 성공
- **MLE**: $\hat\theta_{MLE} = k/n$
- **MAP**: $\hat\theta_{MAP} = (\alpha + k - 1)/(\alpha + \beta + n - 2)$ (Beta mode)
- **Full Bayesian**: posterior = Beta$(\alpha + k, \beta + n - k)$, mean = $(\alpha+k)/(\alpha+\beta+n)$
- Uniform prior Beta$(1,1)$: MLE = MAP (정리 2.1)

**예시 2 — Gaussian 평균 추정**:
- $x_i \sim \mathcal{N}(\mu, \sigma^2)$, prior $\mu \sim \mathcal{N}(0, \tau^2)$
- **MLE**: $\hat\mu_{MLE} = \bar x$
- **MAP**: $\hat\mu_{MAP} = \frac{\tau^2}{\tau^2 + \sigma^2/n}\bar x$ — **shrinkage** toward 0
- **Full Bayesian**: posterior = $\mathcal{N}(\hat\mu_{MAP}, (\sigma^{-2}n + \tau^{-2})^{-1})$

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# ────────────────────────────────────────────────
# Setup: 동전 n=10번 던져 k=7번 앞면
# ────────────────────────────────────────────────
n, k = 10, 7
alpha, beta_ = 2.0, 2.0  # Beta(2,2) prior

theta = np.linspace(1e-3, 1 - 1e-3, 1000)
prior = stats.beta(alpha, beta_).pdf(theta)
likelihood = theta**k * (1 - theta)**(n - k)
posterior = stats.beta(alpha + k, beta_ + n - k).pdf(theta)

# ────────────────────────────────────────────────
# MLE, MAP, Posterior mean
# ────────────────────────────────────────────────
theta_mle = k / n                                        # = 0.7
theta_map = (alpha + k - 1) / (alpha + beta_ + n - 2)    # = 8/12
theta_mean = (alpha + k) / (alpha + beta_ + n)           # = 9/14

print(f"MLE               : {theta_mle:.4f}")
print(f"MAP (Beta(2,2))   : {theta_map:.4f}")
print(f"Posterior mean    : {theta_mean:.4f}")
print(f"Posterior 95% CI  : [{stats.beta(alpha+k, beta_+n-k).ppf(0.025):.4f}, "
      f"{stats.beta(alpha+k, beta_+n-k).ppf(0.975):.4f}]")

# ────────────────────────────────────────────────
# Uniform prior Beta(1,1)에서 MLE = MAP 확인
# ────────────────────────────────────────────────
alpha_u, beta_u = 1.0, 1.0
theta_map_unif = (alpha_u + k - 1) / (alpha_u + beta_u + n - 2)
print(f"\nUniform prior 하에서 MAP = {theta_map_unif:.4f} (= MLE = {theta_mle:.4f})")

# ────────────────────────────────────────────────
# L2 regularization vs Gaussian prior 등가성
# ────────────────────────────────────────────────
rng = np.random.default_rng(0)
X = rng.standard_normal((100, 5))
w_true = np.array([1.0, -0.5, 0.3, 0.0, -0.2])
y = X @ w_true + 0.1 * rng.standard_normal(100)

sigma2, tau2 = 0.01, 1.0
lam = sigma2 / tau2

# Method A: L2-regularized least squares
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(5), X.T @ y)

# Method B: MAP with Gaussian prior (수식으로 동일)
#   posterior precision = X^T X / sigma2 + I / tau2
#   posterior mean = (X^T X/sigma2 + I/tau2)^(-1) X^T y / sigma2
prec = X.T @ X / sigma2 + np.eye(5) / tau2
w_map = np.linalg.solve(prec, X.T @ y / sigma2)

print(f"\nRidge:          {np.round(w_ridge, 4)}")
print(f"Gaussian MAP:   {np.round(w_map, 4)}")
print(f"Difference:     {np.max(np.abs(w_ridge - w_map)):.2e}")

# ────────────────────────────────────────────────
# Plot
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta, posterior, 'k-', lw=2.5, label='Posterior Beta(9,5)')
ax.axvline(theta_mle, color='C0', ls='--', lw=2, label=f'MLE = {theta_mle:.3f}')
ax.axvline(theta_map, color='C1', ls='--', lw=2, label=f'MAP = {theta_map:.3f}')
ax.axvline(theta_mean, color='C2', ls='--', lw=2, label=f'Posterior mean = {theta_mean:.3f}')

ci_low, ci_high = stats.beta(alpha+k, beta_+n-k).ppf([0.025, 0.975])
ax.axvspan(ci_low, ci_high, alpha=0.15, color='gray', label=f'95% CI [{ci_low:.2f},{ci_high:.2f}]')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel('posterior density')
ax.set_title('MLE vs MAP vs Full Bayesian — 세 접근의 출력 비교')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('mle_map_bayes.png', dpi=150); plt.show()
```

**출력**:
```
MLE               : 0.7000
MAP (Beta(2,2))   : 0.6667
Posterior mean    : 0.6429
Posterior 95% CI  : [0.3840, 0.8623]

Uniform prior 하에서 MAP = 0.7000 (= MLE = 0.7000)

Ridge:          [ 0.9832 -0.4821  0.2977  0.0053 -0.1952]
Gaussian MAP:   [ 0.9832 -0.4821  0.2977  0.0053 -0.1952]
Difference:     2.78e-17
```

Ridge와 Gaussian MAP이 **기계정밀도**까지 일치 — 정리 2.2의 수치적 검증.

---

## 🔗 AI/ML 연결

### Weight Decay의 Bayesian 해석

PyTorch의 `optim.SGD(..., weight_decay=lam)`은 loss에 $\frac{\lam}{2}\|W\|^2$를 추가. 이것이 정확히 **Gaussian prior $\mathcal{N}(0, 1/\lambda)$의 MAP**. 학습된 NN은 "MAP estimate of a BNN with isotropic Gaussian prior".

### LASSO와 Sparse Coding

$L_1$ penalty = Laplace prior MAP → sparsity를 선호. 이미지의 sparse coding, compressed sensing의 Bayesian 해석.

### Dropout의 Implicit Prior

Gal & Ghahramani (2016, Ch5-04)는 dropout이 Bernoulli prior의 variational approximation임을 보임. Dropout rate $p$ = prior 의 scale.

### BNN의 MAP vs Full

MAP BNN = 1개 NN(일반 학습된 것). Full BNN = 가중치 분포. **예측 불확실성**을 원하면 full 필요 → Laplace(Ch5-02), BBB(Ch5-03), MC Dropout(Ch5-04).

### Laplace approximation의 위치

Laplace는 "MAP 주변 2차 Taylor"로 posterior를 **Gaussian으로 근사** — MAP의 자연스러운 업그레이드. MAP이 mode만, Laplace는 mode + curvature → unified into Ch5-02.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Prior가 well-specified | Prior가 틀리면 MAP이 잘못된 곳에 집중 |
| $\arg\max$ 유일 | Multimodal posterior는 MAP이 하나 선택 → 큰 정보 손실 |
| 연속 $\Theta$ | Discrete $\theta$는 MAP과 posterior mode가 다를 수 있음 |
| Reparam 불변 불필요 | MAP은 reparam 비불변 (정리 2.5) → 분석에 주의 |
| 계산 가능 $\hat\theta_{MAP}$ | 고차원·비볼록에서 local optima |
| MAP이 posterior를 대표 | Skewed/heavy-tailed posterior에서 mode ≠ mean ≠ median |

**실무 팁**: "MAP만 쓰면 안 되는 경우" = (1) 불확실성이 의사결정에 중요, (2) multimodal posterior, (3) 예측의 calibration이 중요 (Ch7-04).

---

## 📌 핵심 정리

$$\boxed{\text{MLE} \subset \text{MAP} \subset \text{Full Bayesian}}$$

| 접근 | 수식 | 무엇을 얻는가 |
|------|------|---------------|
| **MLE** | $\arg\max \log p(D\|\theta)$ | 점추정 |
| **MAP** | $\arg\max [\log p(D\|\theta) + \log p(\theta)]$ | 점추정 + prior 정보 |
| **Full Bayesian** | $p(\theta\|D) \propto p(D\|\theta)p(\theta)$ | 분포 전체 |

핵심 등식:
- **L2 reg = Gaussian prior MAP**, $\lambda = \sigma^2/\tau^2$
- **L1 reg = Laplace prior MAP**
- **MLE = Uniform prior의 MAP** (improper prior 주의)
- **MAP = Delta-posterior의 full Bayesian**

---

## 🤔 생각해볼 문제

**문제 1** (기초): Beta$(\alpha, \beta)$ prior 하에서 MAP과 posterior mean이 언제 같은가?

<details>
<summary>힌트 및 해설</summary>

Beta$(\alpha+k, \beta+n-k)$의 mode = $(\alpha+k-1)/(\alpha+\beta+n-2)$, mean = $(\alpha+k)/(\alpha+\beta+n)$.

두 값이 같으려면 $\frac{\alpha+k-1}{\alpha+\beta+n-2} = \frac{\alpha+k}{\alpha+\beta+n}$.

정리: $(\alpha+k-1)(\alpha+\beta+n) = (\alpha+k)(\alpha+\beta+n-2)$
$\Rightarrow \alpha+\beta+n = 2(\alpha+k)$

즉 $\alpha = k, \beta = n - k$ (Beta가 대칭으로 likelihood를 두 배 "본 것"과 같음) 또는 일반적으로 posterior가 **대칭(symmetric)**일 때. 실전적으로 MAP ≠ mean이 일반적이며, skewness가 클수록 차이 커짐.

</details>

**문제 2** (심화): "MLE가 reparam 불변"을 증명하고, 왜 MAP은 불변이 아닌지 구체적 예로 보여라.

<details>
<summary>힌트 및 해설</summary>

**MLE 불변**: one-to-one $\phi = g(\theta)$에서 $p(D|\phi) = p(D|g^{-1}(\phi))$. 따라서 $\arg\max_\phi p(D|\phi) = g(\arg\max_\theta p(D|\theta))$. $\square$

**MAP 비불변 예시**:
- $\theta \in (0,1)$, prior uniform on $\theta$: $p(\theta) = 1$
- Reparam $\phi = \log(\theta/(1-\theta))$ (logit): $p_\phi(\phi) = \frac{e^\phi}{(1+e^\phi)^2}$ (Jacobian 있음)

$\theta$-공간에서 MAP은 uniform이므로 likelihood argmax, 즉 MLE = $k/n$.
$\phi$-공간에서 MAP은 prior가 $\phi = 0$에서 peak인 density이므로 MAP이 MLE와 달라짐 — **좌표 변환이 MAP 위치를 바꿈**.

Full Bayesian은 분포를 변환해서 다루므로 이런 일이 없다.

</details>

**문제 3** (AI 연결): PyTorch에서 `weight_decay=1e-4`를 설정하면 이것은 어떤 Gaussian prior의 MAP인가? 학습률 $\eta$와의 관계는?

<details>
<summary>힌트 및 해설</summary>

PyTorch weight_decay $\lambda$는 **loss에 $\lambda \|w\|^2 / 2$ 추가**(일부 구현은 $\lambda \|w\|^2$). 이것이 Gaussian prior $\mathcal{N}(0, 1/\lambda)$의 로그($-\lambda\|w\|^2/2$)의 음수에 대응.

즉 **prior variance = $1/\lambda$**. $\lambda = 10^{-4}$이면 $\tau^2 = 10^4$, 즉 표준편차 100인 Gaussian prior.

**학습률과는 분리**: $\eta$는 optimization step size, $\lambda$는 prior strength. 단, "$w \leftarrow (1 - \eta\lambda)w - \eta\nabla L$" 업데이트 규칙에서 두 값이 곱으로 나타남 → effective decay per step = $\eta\lambda$.

**BNN 해석**: 이 MAP은 full BNN에서 얻는 posterior의 mode를 찾는 것 — 불확실성 없음. MC Dropout(Ch5-04)이나 Laplace(Ch5-02)로 업그레이드 가능.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Bayes 정리와 4가지 역할](./01-bayes-rule-four-roles.md) | [📚 README로 돌아가기](../README.md) | [03. Conjugate Priors의 수학 ▶](./03-conjugate-priors.md) |

</div>
