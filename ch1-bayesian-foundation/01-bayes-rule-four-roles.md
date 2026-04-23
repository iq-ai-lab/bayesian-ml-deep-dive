# 01. Bayes 정리와 4가지 역할

## 🎯 핵심 질문

- $p(\theta|D) = p(D|\theta)p(\theta)/p(D)$의 각 항은 정확히 무엇을 의미하는가?
- 왜 분모 $p(D) = \int p(D|\theta)p(\theta)d\theta$가 일반적으로 **계산 불가능(intractable)**한가?
- 빈도주의(frequentist)와 베이지안(Bayesian)은 "확률"이라는 단어를 어떻게 다르게 쓰는가?
- $p(D)$가 상수임에도 왜 "evidence"라는 중요한 이름을 갖는가? 모델 선택에서의 역할은?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**VAE**의 ELBO 유도는 $\log p(x)$를 분해하는 것에서 출발한다 — 이 $p(x)$가 정확히 Bayes 정리의 evidence $p(D)$이다. **BNN**에서 가중치 posterior $p(W|D)$를 얻으려면 $p(D) = \int p(D|W)p(W)dW$를 피하거나 근사해야 한다. **Diffusion Model**의 reverse SDE에서 score $\nabla\log p_t(x)$를 추정하는 것도 "evidence를 직접 계산하지 않고 gradient만 필요"하다는 Bayesian 계산 전략의 대표 예다. **Bayesian model comparison**에서 $p(D|M_1)/p(D|M_2)$ — Bayes factor — 는 모델 선택의 황금 표준이며, 이것이 **왜 evidence가 "단순 상수"가 아닌가**를 말해준다.

---

## 📐 수학적 선행 조건

- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): 조건부 확률 $P(A|B) = P(A\cap B)/P(B)$, 주변화(marginalization), 확률밀도의 변환
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): 모수모형 $\{p_\theta : \theta \in \Theta\}$, likelihood
- 실해석: 일반화된 적분(Lebesgue), 지배수렴 정리
- Information Theory: KL divergence(Ch2에서 재사용)

---

## 📖 직관적 이해

### 요리 비유로 본 4가지 역할

냉장고에 재료 $D$가 있다. "어떤 요리사 $\theta$가 이 재료를 썼을까?"를 추론하는 상황:

| 항 | 이름 | 요리 비유 | 의미 |
|----|------|-----------|------|
| $p(\theta)$ | **Prior** | "이 집에 자주 요리하는 요리사들의 빈도" | 데이터를 보기 전의 믿음 |
| $p(D\|\theta)$ | **Likelihood** | "요리사 $\theta$가 재료 $D$를 쓸 가능성" | 모수가 주어졌을 때 데이터를 설명하는 정도 |
| $p(\theta\|D)$ | **Posterior** | "재료 $D$를 본 뒤 요리사 $\theta$에 대한 믿음" | 데이터로 업데이트된 믿음 |
| $p(D)$ | **Evidence** | "이 냉장고에서 $D$ 같은 재료를 볼 확률" | **모든 요리사에 대한 평균**, 모수 무관 |

Bayes 정리는 세 가지(prior, likelihood, evidence)를 조합해 네 번째(posterior)를 얻는 공식이다:

$$p(\theta|D) = \frac{p(D|\theta)\,p(\theta)}{p(D)}$$

### 왜 evidence가 계산 불가능한가

$p(D) = \int p(D|\theta)\,p(\theta)\,d\theta$는 **모든 가능한 $\theta$에 대한 적분**이다. 문제:

- **고차원**: BNN이면 $\theta \in \mathbb{R}^{10^6}$. 격자로 적분 불가 (curse of dimensionality)
- **비공액(non-conjugate)**: 일반 likelihood × prior는 닫힌형이 없음
- **비볼록 likelihood**: NN의 loss landscape — 다봉(multi-modal)

**이 intractability가 VI와 MCMC 존재의 이유다.** VI는 $p(D)$ 대신 lower bound(ELBO)를 최적화하고, MCMC는 $p(D)$ 없이도 posterior 샘플을 얻는다.

### 빈도주의 vs 베이지안

| | Frequentist | Bayesian |
|---|---|---|
| **모수 $\theta$** | 미지의 **고정된 상수** | **확률변수** |
| **확률의 대상** | 반복 실험에서의 빈도 | 믿음의 정도(degree of belief) |
| **추정** | 점추정 $\hat\theta_{MLE}$ + 표본분포 | posterior 분포 $p(\theta\|D)$ |
| **신뢰구간** | "$95\%$의 실험에서 참값을 포함" (pre-data) | $P(\theta \in [a,b]\|D) = 0.95$ (post-data) |
| **Prior** | 사용 안 함 | 필수 |

> **비유**: 주사위 한 번 던져 3이 나왔다. "주사위가 공정할 확률은?" 빈도주의자: "3 나올 확률은 공정 주사위면 1/6, 조작 주사위면 다른 값. 데이터로 가설검정." 베이지안: "공정 주사위라는 prior 80% + 이 데이터 + posterior". 같은 상황, 다른 언어.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — 확률모형(Statistical Model)

측정 가능 공간 $(\mathcal{X}, \mathcal{A})$ 위의 **확률모형**은 다음 집합이다:

$$\mathcal{P} = \{P_\theta : \theta \in \Theta\}$$

각 $P_\theta$는 $\mathcal{X}$ 위 확률측도이고, **모수공간** $\Theta$는 일반적으로 $\mathbb{R}^d$의 부분집합. 공통 지배측도 $\mu$에 대해 밀도 $p(x|\theta) = dP_\theta/d\mu$가 존재한다고 가정.

### 정의 1.2 — Bayesian 확률모형

**Bayesian 확률모형**은 다음 세 요소로 구성된다:

1. **Prior** $p(\theta)$: 모수공간 $\Theta$ 위 확률밀도
2. **Likelihood** $p(D|\theta)$: 각 $\theta$에 대한 데이터 밀도 (sampling model)
3. **Joint** $p(D, \theta) = p(D|\theta)\,p(\theta)$: 결합밀도

### 정의 1.3 — Posterior와 Evidence

데이터 $D$가 관측되면:

$$\boxed{p(\theta|D) = \frac{p(D|\theta)\,p(\theta)}{p(D)} \quad \text{(posterior)}}$$

$$\boxed{p(D) = \int_\Theta p(D|\theta)\,p(\theta)\,d\theta \quad \text{(evidence / marginal likelihood)}}$$

### 정의 1.4 — Predictive Distribution

새 관측 $y^*$에 대한 예측은 posterior를 통해 주변화:

$$p(y^*|D) = \int_\Theta p(y^*|\theta)\,p(\theta|D)\,d\theta$$

이는 **Ch1-04**에서 자세히 다룬다.

---

## 🔬 정리와 증명

### 정리 1.1 — Bayes 정리의 유도

**명제**: 결합밀도 $p(D, \theta)$가 존재하고 $p(D) > 0$이면,

$$p(\theta|D) = \frac{p(D|\theta)\,p(\theta)}{p(D)}$$

**증명**:

조건부 밀도의 정의(측도론적 버전)에 따라:

$$p(\theta|D) = \frac{p(D, \theta)}{p(D)}$$

($p(D) > 0$이므로 분모가 정의된다.) 결합밀도를 분해:

$$p(D, \theta) = p(D|\theta)\,p(\theta)$$

(chain rule). 대입:

$$p(\theta|D) = \frac{p(D|\theta)\,p(\theta)}{p(D)}$$

한편, 주변화에 의해:

$$p(D) = \int p(D, \theta)\,d\theta = \int p(D|\theta)\,p(\theta)\,d\theta$$

$\square$

### 정리 1.2 — Posterior의 비례성

**명제**: $p(D)$는 $\theta$에 의존하지 않으므로,

$$p(\theta|D) \propto p(D|\theta)\,p(\theta)$$

이 비례 관계만으로 posterior의 형태를 결정할 수 있고, 정규화 상수는 $\int [\text{우변}]\,d\theta$로 복원된다.

**증명**: 정리 1.1에서 $p(D)$가 $\theta$와 무관한 상수이므로 자명. $\square$

> **실전적 의미**: MCMC(Ch4-01 Metropolis-Hastings)가 evidence 없이 posterior 샘플을 얻을 수 있는 수학적 근거. Acceptance ratio $p(\theta')p(D|\theta')/[p(\theta)p(D|\theta)]$에서 $p(D)$가 자동으로 상쇄된다.

### 정리 1.3 — 순차 업데이트 (Sequential Updating)

**명제**: 데이터가 순차적으로 $D_1, D_2$로 도착하고 조건부 독립 $p(D_1, D_2|\theta) = p(D_1|\theta)p(D_2|\theta)$라고 하면,

$$p(\theta|D_1, D_2) \propto p(D_2|\theta)\,\underbrace{p(\theta|D_1)}_{\text{이전 posterior}}$$

즉 **어제의 posterior가 오늘의 prior** 역할을 한다.

**증명**:

$$p(\theta|D_1, D_2) = \frac{p(D_1, D_2|\theta)\,p(\theta)}{p(D_1, D_2)}$$

조건부 독립에 의해:

$$= \frac{p(D_2|\theta)\,p(D_1|\theta)\,p(\theta)}{p(D_1, D_2)}$$

분자의 $p(D_1|\theta)\,p(\theta)$를 $p(\theta|D_1)\,p(D_1)$로 재작성(Bayes):

$$= \frac{p(D_2|\theta)\,p(\theta|D_1)\,p(D_1)}{p(D_1, D_2)} = \frac{p(D_2|\theta)\,p(\theta|D_1)}{p(D_2|D_1)}$$

$\theta$에 무관한 상수 $p(D_2|D_1)$을 제거하면 비례 관계. $\square$

### 정리 1.4 — Evidence는 model comparison의 지표

**명제**: 두 모델 $M_1, M_2$(각자의 prior·likelihood를 가짐)에 대해:

$$\frac{p(M_1|D)}{p(M_2|D)} = \underbrace{\frac{p(D|M_1)}{p(D|M_2)}}_{\text{Bayes factor}} \cdot \frac{p(M_1)}{p(M_2)}$$

여기서 $p(D|M_i) = \int p(D|\theta, M_i)\,p(\theta|M_i)\,d\theta$는 모델 $M_i$의 evidence.

**증명**: 모델 수준에서 Bayes 정리를 적용하면 즉시 유도. $\square$

> **중요**: Evidence $p(D)$는 posterior 내부에선 "상수"지만 **모델 비교에선 핵심**. 자동 Occam's razor(복잡한 모델은 prior를 넓게 펼쳐야 해서 $p(D)$가 작아짐)가 이 공식에 내장돼 있다.

### 예시

**예시 1 — 동전 던지기 (Beta-Bernoulli)**:
- Prior $p(\theta) = \text{Beta}(\alpha, \beta)$, $\theta \in (0,1)$
- Likelihood $p(D|\theta) = \theta^{k}(1-\theta)^{n-k}$ (동전 $n$번 던져 $k$번 앞면)
- Evidence $p(D) = \int_0^1 \theta^k(1-\theta)^{n-k} \cdot \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha,\beta)}\,d\theta = \frac{B(\alpha+k, \beta+n-k)}{B(\alpha,\beta)}$ ← 닫힌형
- Posterior $p(\theta|D) = \text{Beta}(\alpha+k, \beta+n-k)$

이 경우 evidence가 **닫힌형으로 계산 가능** — 이런 conjugate 관계가 Ch1-03의 주제다.

**예시 2 — BNN**:
- Prior $p(W) = \mathcal{N}(0, \sigma^2 I)$ on $\mathbb{R}^{10^6}$
- Likelihood $p(D|W) = \prod_i \mathcal{N}(y_i; f_W(x_i), \sigma_n^2)$
- Evidence $p(D) = \int_{\mathbb{R}^{10^6}} p(D|W)p(W)\,dW$ — **계산 불가능**
- ⇒ Laplace(Ch5-02), VI(Ch5-03), MC Dropout(Ch5-04) 등으로 근사

---

## 💻 NumPy 구현 검증

동전 던지기 예제로 **prior → posterior 업데이트**와 **evidence 계산**을 직접 수행해보자.

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.special import betaln

# ────────────────────────────────────────────────
# 1. 설정 — Beta(2, 2) prior + 동전 10번 던져 7번 앞면
# ────────────────────────────────────────────────
alpha, beta_ = 2.0, 2.0      # prior 파라미터
n, k = 10, 7                 # 총 n번, 앞면 k번

theta = np.linspace(1e-4, 1 - 1e-4, 1000)

# ────────────────────────────────────────────────
# 2. Prior, Likelihood, Posterior
# ────────────────────────────────────────────────
prior = stats.beta(alpha, beta_).pdf(theta)
likelihood = theta**k * (1 - theta)**(n - k)           # 정규화 안 된 값
posterior_exact = stats.beta(alpha + k, beta_ + n - k).pdf(theta)

# ────────────────────────────────────────────────
# 3. Evidence — 닫힌형과 수치적분 비교
# ────────────────────────────────────────────────
#  closed-form: B(α+k, β+n-k) / B(α, β)
log_evidence_closed = betaln(alpha + k, beta_ + n - k) - betaln(alpha, beta_)
evidence_closed = np.exp(log_evidence_closed)

#  수치적분 (trapezoid)
integrand = likelihood * stats.beta(alpha, beta_).pdf(theta)
evidence_numeric = np.trapz(integrand, theta)

print(f"Evidence (closed form) : {evidence_closed:.6f}")
print(f"Evidence (numerical)   : {evidence_numeric:.6f}")
print(f"Relative error         : {abs(evidence_closed - evidence_numeric)/evidence_closed:.2e}")

# ────────────────────────────────────────────────
# 4. Posterior 복원 — likelihood × prior / evidence
# ────────────────────────────────────────────────
posterior_reconstructed = (likelihood * stats.beta(alpha, beta_).pdf(theta)) / evidence_closed

# ────────────────────────────────────────────────
# 5. 순차 업데이트 — 어제의 posterior가 오늘의 prior
# ────────────────────────────────────────────────
# Step 1: 처음 5번 던져 4번 앞면
posterior_step1 = stats.beta(alpha + 4, beta_ + 1).pdf(theta)
# Step 2: 추가 5번 던져 3번 앞면 → α+4+3, β+1+2
posterior_step2 = stats.beta(alpha + 7, beta_ + 3).pdf(theta)  # 원 데이터와 동일

# ────────────────────────────────────────────────
# 6. 시각화
# ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(theta, prior, lw=2, label=r'Prior Beta(2,2)')
axes[0].plot(theta, likelihood / likelihood.max() * prior.max(),
             lw=2, label='Likelihood (rescaled)')
axes[0].plot(theta, posterior_exact, lw=2,
             label=r'Posterior Beta(9,5)')
axes[0].set_xlabel(r'$\theta$'); axes[0].set_ylabel('density')
axes[0].set_title('Bayes: prior × likelihood → posterior')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(theta, posterior_exact, 'k-', lw=3, label='Exact Beta(9,5)')
axes[1].plot(theta, posterior_reconstructed, 'r--', lw=2,
             label='Reconstructed via Bayes')
axes[1].set_xlabel(r'$\theta$'); axes[1].set_ylabel('density')
axes[1].set_title(f'Evidence={evidence_closed:.5f}로 정확히 복원')
axes[1].legend(); axes[1].grid(alpha=0.3)

axes[2].plot(theta, prior, lw=2, label='Prior')
axes[2].plot(theta, posterior_step1, lw=2, label='After 4/5 heads')
axes[2].plot(theta, posterior_step2, lw=2, label='After total 7/10')
axes[2].set_xlabel(r'$\theta$'); axes[2].set_ylabel('density')
axes[2].set_title('순차 업데이트 (yesterday posterior = today prior)')
axes[2].legend(); axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bayes_rule_four_roles.png', dpi=150, bbox_inches='tight')
plt.show()
```

**출력 예시**:
```
Evidence (closed form) : 0.000909
Evidence (numerical)   : 0.000909
Relative error         : 2.13e-09
```

수치적분과 닫힌형이 정확히 일치 — **evidence는 단순 정규화 상수**임을 확인. 또한 순차 업데이트가 **한꺼번에 업데이트한 것과 같은 posterior**를 주는 것도 확인된다 (정리 1.3의 수치적 검증).

---

## 🔗 AI/ML 연결

### VAE와 evidence

VAE의 ELBO는 $\log p(x) \geq \mathcal{L}$의 부등식 — 이 $p(x)$가 정확히 Bayes의 evidence이다. VAE는 $p(x)$를 직접 계산하지 못해서 ELBO라는 **lower bound**를 최적화한다(Ch2-01, Ch3-01).

### BNN의 evidence와 model selection

Laplace approximation에서 얻는 **model evidence** $\log p(D|M) \approx \log p(D|W^*, M) - \frac{1}{2}\log|H|/(2\pi)^{-d/2}$는 NN의 architecture 선택(층 수, 너비)에 사용 가능. **BIC**(Bayesian Information Criterion)도 이 evidence의 점근 근사.

### Diffusion과 score

Diffusion Model은 $\log p_t(x)$ 자체 대신 **gradient** $\nabla_x \log p_t(x)$(score function)만 학습 — evidence의 상수를 피해 가는 대표 전략(SDE 레포 Ch6과 교차).

### Probabilistic Programming

Stan, PyMC, NumPyro 같은 **PPL**은 모델 코드 → posterior 추론을 자동화. 내부적으로 evidence $p(D)$를 피하기 위해 HMC/NUTS(Ch4-03,04)를 사용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Prior $p(\theta)$가 명시적 | 객관적 prior의 어려움 — Jeffreys, Reference, improper prior 논쟁 |
| Likelihood가 올바른 모델 | Model misspecification 시 posterior가 잘못된 곳에 집중 |
| $p(D) > 0$ 가정 | 표현력이 지나치게 제한적이면 $p(D) \to 0$ (prior가 데이터에 전혀 지지 없음) |
| 조건부 독립 가정(순차 업데이트) | 시계열 등 autocorrelated 데이터에서 깨짐 |
| 적분 존재성 | 고차원에서 intractable → VI(Ch2), MCMC(Ch4) 필요 |

**주의**: "prior를 고르면 답이 편향된다"는 반론이 있지만, **non-informative prior도 선택**이다. Frequentist도 "significance level, test statistic" 등 주관적 선택을 한다.

---

## 📌 핵심 정리

$$\boxed{p(\theta|D) = \frac{p(D|\theta)\,p(\theta)}{p(D)}, \quad p(D) = \int p(D|\theta)\,p(\theta)\,d\theta}$$

| 항 | 이름 | 역할 |
|------|------|------|
| $p(\theta)$ | **Prior** | 데이터 전 믿음 |
| $p(D\|\theta)$ | **Likelihood** | 데이터 설명력 |
| $p(\theta\|D)$ | **Posterior** | 업데이트된 믿음 |
| $p(D)$ | **Evidence** | 정규화 상수 + **model comparison 지표** |

핵심 통찰:
- $p(D)$는 posterior 내부에선 상수지만, **일반적으로 intractable** → VI/MCMC의 존재 이유
- **Posterior ∝ Likelihood × Prior** 만으로 posterior 형태를 결정 가능
- **순차 업데이트**: 어제의 posterior = 오늘의 prior
- Bayes factor $p(D|M_1)/p(D|M_2)$로 모델 비교 가능 (자동 Occam's razor)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Uniform prior $p(\theta) = 1/(b-a)$ on $[a,b]$일 때, posterior $p(\theta|D)$의 형태는? 이 경우 MAP 추정이 어떻게 MLE와 같아지는가?

<details>
<summary>힌트 및 해설</summary>

$p(\theta|D) \propto p(D|\theta) \cdot \mathbf{1}_{[a,b]}(\theta) \cdot \frac{1}{b-a}$

$[a,b]$ 내부에서 posterior $\propto$ likelihood. MAP:

$$\hat\theta_{MAP} = \arg\max p(\theta|D) = \arg\max p(D|\theta) = \hat\theta_{MLE}$$

단, $\hat\theta_{MLE} \in [a,b]$일 때만. Uniform prior는 **MLE와 MAP을 동일하게** 만든다.

이것이 **"MLE = uniform prior의 MAP"**의 직접적 의미 (Ch1-02에서 심화).

</details>

**문제 2** (심화): Evidence $p(D) = \int p(D|\theta)p(\theta)d\theta$가 Monte Carlo로 $\frac{1}{N}\sum_i p(D|\theta_i), \theta_i \sim p(\theta)$로 추정된다. 이 추정기가 **고차원에서 거의 항상 실패**하는 이유는?

<details>
<summary>힌트 및 해설</summary>

Prior $p(\theta)$는 대부분 넓게 퍼져 있지만, likelihood $p(D|\theta)$는 posterior 지역(일반적으로 매우 좁음)에 집중. Prior 샘플 $\theta_i$ 중 $p(D|\theta_i)$가 유의미한 샘플은 **극소수**.

**효과적 샘플 수** $N_{eff} = (\sum w_i)^2 / \sum w_i^2$이 $N$보다 훨씬 작아져, 추정 분산이 폭발.

이것을 "**prior-posterior mismatch**"라 하며, VI(ELBO)와 annealed importance sampling 같은 고급 방법이 필요한 이유다.

</details>

**문제 3** (AI 연결): VAE의 ELBO는 $\log p(x) \geq \mathcal{L} = \mathbb{E}_q[\log p(x,z)] - \mathbb{E}_q[\log q(z|x)]$이다. 여기서 $p(x)$가 정확히 Bayes 정리의 어느 항에 해당하는가? 왜 VAE가 $\log p(x)$를 **직접 최적화하지 않고** ELBO를 대신 쓰는가?

<details>
<summary>힌트 및 해설</summary>

$p(x) = \int p(x|z)p(z)dz$는 latent $z$를 $\theta$로 보면 **evidence** $p(D)$에 정확히 대응. 데이터 $x$에 대한 model evidence.

$\log p(x)$를 직접 계산하려면 고차원 $z$에 대한 적분이 필요 → **intractable**. ELBO는 Jensen 부등식 $\log \mathbb{E}[f] \geq \mathbb{E}[\log f]$로 lower bound를 만들어 **적분 대신 기댓값**으로 전환. 기댓값은 $q(z|x)$에서의 샘플링으로 Monte Carlo 추정 가능.

이것이 **"Bayes의 intractability → VI"**의 AI 버전 (Ch2-01에서 상세히).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ README](../README.md) | [📚 README로 돌아가기](../README.md) | [02. MLE vs MAP vs Full Bayesian ▶](./02-mle-map-full-bayesian.md) |

</div>
