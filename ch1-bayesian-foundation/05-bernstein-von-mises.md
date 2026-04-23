# 05. Posterior의 점근 성질 (Bernstein–von Mises)

## 🎯 핵심 질문

- 데이터 $n \to \infty$에서 posterior는 어떤 모양으로 수렴하는가?
- 왜 posterior가 **$\mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)$**로 수렴하는가?
- BvM 정리의 **regularity 조건**은 정확히 무엇인가?
- 대규모 데이터에서 **Bayesian과 Frequentist가 같은 답**을 주는 이유는?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**"Prior 선택이 큰 문제 아닌가?"**에 대한 답이 여기 있다. 데이터가 충분하면 prior 영향이 **소멸**하고 posterior가 Fisher 정보 기반 Gaussian으로 수렴. 이것이 **Laplace approximation**(Ch5-02)의 이론적 근거 — BvM이 성립하면 Laplace가 asymptotically exact. **Bayesian credible interval** $[a, b]$와 **Frequentist confidence interval**이 수렴하는 것도 BvM의 직접 결과. 이것이 대규모 BNN에서 posterior가 사실상 Gaussian처럼 행동하는 이유(실험적 관찰).

---

## 📐 수학적 선행 조건

- [Ch1-01~04](./01-bayes-rule-four-roles.md): Posterior, conjugate, predictive
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): MLE 점근 정규성, Fisher 정보, CLT
- [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive): Total variation distance, weak convergence
- 실해석: Taylor 전개, uniform convergence

---

## 📖 직관적 이해

### BvM의 선언

"**충분한 데이터가 있으면 posterior는 MLE 주변 Gaussian과 구분 불가능하고, prior는 잊혀진다**."

$$p(\theta|D_n) \xrightarrow{\text{TV}} \mathcal{N}\left(\hat\theta_{MLE}, \frac{1}{n}F^{-1}(\theta_0)\right)$$

- $\theta_0$: 참값(존재 가정)
- $F(\theta_0)$: Fisher 정보 행렬
- $\hat\theta_{MLE}$: 샘플 의존 (확률변수)
- TV: total variation distance

### 네 가지 귀결

1. **Prior가 잊혀짐**: prior의 선택이 posterior shape에 거의 영향 없음 ($n$이 크면)
2. **Credible = Confidence**: Bayesian 95% credible interval ≈ frequentist 95% CI
3. **Posterior는 Gaussian**: 정확한 posterior 대신 **Laplace(Gaussian 근사)**로 충분
4. **수렴률 $O(n^{-1/2})$**: SEM scales like $1/\sqrt{n}$ — CLT와 동일

### 요리 비유

- **소표본**: 요리사에 대한 "개성 있는" prior 믿음이 결과에 반영 ("이 동네는 김철수가 자주 온다")
- **대표본 수천 관측**: prior는 증발, "객관적" 추정만 남음. Bayesian이든 frequentist든 같은 답.

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Total Variation Distance

두 확률측도 $P, Q$에 대해:

$$d_{TV}(P, Q) = \sup_A |P(A) - Q(A)| = \frac{1}{2}\int |p - q|\,d\mu$$

$d_{TV}(P, Q) \to 0$이면 **강한** 수렴(weak convergence보다 강함).

### 정의 5.2 — Fisher Information

모델 $p(x|\theta)$, $\theta \in \Theta \subseteq \mathbb{R}^d$에 대해:

$$F(\theta) = -\mathbb{E}\left[\nabla^2 \log p(x|\theta)\right] = \mathbb{E}\left[\nabla \log p(x|\theta) (\nabla \log p(x|\theta))^T\right]$$

$d \times d$ 행렬. MLE의 점근 분산을 결정 — Cramér-Rao lower bound.

### 정의 5.3 — Regularity Conditions

BvM이 성립하기 위한 표준 조건들(비형식):

1. **Identifiability**: $\theta_1 \neq \theta_2 \Rightarrow p(\cdot|\theta_1) \neq p(\cdot|\theta_2)$
2. **Smooth likelihood**: $\log p(x|\theta)$가 $\theta$에 대해 $C^3$
3. **Fisher information positive definite**: $F(\theta_0) \succ 0$
4. **Interior MLE**: $\theta_0 \in \text{int}(\Theta)$ (경계 아님)
5. **Prior continuity**: $p(\theta)$가 $\theta_0$에서 연속이고 $p(\theta_0) > 0$
6. **Prior positivity**: 모든 $\theta_0$ 근방에서 $p(\theta) > 0$

---

## 🔬 정리와 증명

### 정리 5.1 — Bernstein–von Mises (1차 버전)

**명제**: 위 regularity 조건 하에서, 참값 $\theta_0$ 주변 IID 샘플이 있을 때:

$$d_{TV}\left(p(\theta|D_n),\ \mathcal{N}\left(\hat\theta_{MLE},\ \frac{F^{-1}(\theta_0)}{n}\right)\right) \xrightarrow{P_{\theta_0}} 0 \quad (n \to \infty)$$

즉 posterior와 $\hat\theta_{MLE}$ 중심 Gaussian 사이 total variation 거리가 참값 하 확률수렴.

**증명 스케치** (Laplace-type):

1. **Re-centering**: $u = \sqrt n(\theta - \hat\theta_{MLE})$로 재매개변수화.

2. **Log-posterior Taylor**:
$$\log p(\theta|D_n) = \log p(\theta) + \sum_i \log p(x_i|\theta) - \log p(D_n)$$

$\hat\theta_{MLE}$ 주변 2차 Taylor($\nabla \ell_n(\hat\theta_{MLE}) = 0$):
$$\sum_i \log p(x_i|\theta) = \ell_n(\hat\theta_{MLE}) - \frac{1}{2}(\theta - \hat\theta_{MLE})^T H_n (\theta - \hat\theta_{MLE}) + R_n$$

$H_n = -\nabla^2 \ell_n(\hat\theta_{MLE})/n$ → Fisher $F(\theta_0)$ by LLN.

3. **Prior의 smoothness**: $\log p(\theta) = \log p(\hat\theta_{MLE}) + O(1/\sqrt n)$ around $\hat\theta_{MLE}$.

4. **Normalization**: 
$$p(\theta|D_n) \propto \exp\left(-\frac{n}{2}(\theta - \hat\theta_{MLE})^T F(\theta_0)(\theta - \hat\theta_{MLE})\right) \cdot (1 + o(1))$$

Gaussian density. Remainder $R_n = o_P(1)$ in appropriate sense.

5. **TV 수렴**: Log-density의 uniform convergence on compacts + tail control → $d_{TV}(p(\theta|D_n), \mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)) \to 0$. $\square$

> **상세 증명**: Van der Vaart "Asymptotic Statistics" Ch10, Ghosh-Ramamoorthi "Bayesian Nonparametrics" Ch4. 비형식 버전은 Gelman BDA3 Appendix B.

### 정리 5.2 — Credible ↔ Confidence Interval의 점근 일치

**명제**: BvM 조건 하에서, $(1-\alpha)$-credible interval과 $(1-\alpha)$-confidence interval이 점근적으로 일치:

$$P_{\theta_0}\left(\theta_0 \in [q_{\alpha/2}^{post}, q_{1-\alpha/2}^{post}]\right) \to 1 - \alpha$$

**증명**: Credible interval = posterior quantile의 구간. BvM으로 posterior가 Gaussian $\mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)$이므로 quantile이 $\hat\theta_{MLE} \pm z_{\alpha/2}\sqrt{F^{-1}/n}$. 이것이 정확히 frequentist CI의 공식(MLE + asymptotic SE). $\square$

> **실전적 의미**: 대규모에서 **Bayesian 신뢰구간이 frequentist를 재현**. 프리퀀시스트 property가 free로 따라옴(단, regularity 조건 하).

### 정리 5.3 — Prior의 점근 무관성

**명제**: $p_1(\theta), p_2(\theta)$ 모두 $\theta_0$ 근방에서 continuous & positive이면:

$$d_{TV}(p_1(\theta|D_n),\ p_2(\theta|D_n)) \xrightarrow{P} 0$$

**증명**: 두 posterior 모두 같은 Gaussian $\mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)$으로 수렴 (정리 5.1). 삼각부등식:
$$d_{TV}(p_1^{post}, p_2^{post}) \leq d_{TV}(p_1^{post}, \mathcal{N}) + d_{TV}(p_2^{post}, \mathcal{N}) \to 0$$

$\square$

**중요 귀결**: 다른 prior를 선택해도 대규모에서 거의 같은 posterior → **prior 감도 낮음**.

### 정리 5.4 — 수렴률과 Contraction Rate

**명제**: BvM 조건 하에서, posterior가 $\theta_0$ 주변 **$n^{-1/2}$-neighborhood**에 집중:

$$P_{\theta_0}\left(p\left(\|\theta - \theta_0\| > M_n/\sqrt{n}\ \Big|\ D_n\right) \to 0\right) \to 1$$

for any $M_n \to \infty$.

**증명**: Gaussian 꼬리가 $\exp(-n u^2 / 2)$으로 decay, $u = \sqrt n (\theta - \theta_0)$. $\square$

### 정리 5.5 — BvM 위반: 비정칙 모델

**명제**: 다음 경우 BvM이 실패할 수 있다:

1. **Non-identifiable**: $\theta_1 \neq \theta_2$인데 $p(\cdot|\theta_1) = p(\cdot|\theta_2)$ — Mixture model의 label switching
2. **Boundary**: $\theta_0 \in \partial\Theta$ — half-normal 한쪽으로만
3. **High-dimensional**: $d = d(n) \to \infty$ — BNN 같은 고차원, 정리 자체가 성립 안 함
4. **Infinite-dim**: Nonparametric Bayes (BvM의 "nonparametric" 확장은 별도 연구)

**귀결**: BvM은 **유한차원, 정칙** 모델에서만. BNN에 대해선 엄밀한 BvM이 성립하지 않고 "effective" Gaussian behavior만 empirical.

### 예시

**예시 1 — Beta-Bernoulli BvM**:

Posterior Beta$(\alpha+k, \beta+n-k)$. $n \to \infty$, $k/n \to \theta_0$:
- Mean = $(\alpha+k)/(\alpha+\beta+n) \to \theta_0$
- Variance = $(\alpha+k)(\beta+n-k)/[(\alpha+\beta+n)^2(\alpha+\beta+n+1)] \approx \theta_0(1-\theta_0)/n$
- Fisher $F(\theta_0) = 1/(\theta_0(1-\theta_0))$ → $F^{-1}/n = \theta_0(1-\theta_0)/n$ ✓

**Prior가 다르면?** Beta(1,1), Beta(0.5, 0.5), Beta(10, 10) 모두 $n \to \infty$에서 같은 Gaussian으로 수렴.

**예시 2 — Normal mean (known variance)**:

Posterior $\mathcal{N}(\mu_n, \tau_n^2)$, $\mu_n = \tau_n^2(\mu_0/\tau_0^2 + n\bar x/\sigma^2)$, $\tau_n^2 = (1/\tau_0^2 + n/\sigma^2)^{-1}$.

$n \to \infty$:
- $\mu_n \to \bar x$ (= MLE)
- $\tau_n^2 \to \sigma^2/n$ (= Fisher inverse / n for Gaussian)

$\square$

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Beta-Bernoulli에서 BvM 수렴 관찰
# ────────────────────────────────────────────────
theta_true = 0.4
ns = [10, 50, 200, 1000, 5000]
priors = [(1, 1), (10, 10), (0.5, 0.5)]  # 세 다른 prior

fig, axes = plt.subplots(1, len(ns), figsize=(20, 4), sharey=True)
theta_grid = np.linspace(0, 1, 500)

for idx, n in enumerate(ns):
    data = rng.binomial(1, theta_true, size=n)
    k = data.sum()
    theta_mle = k / n
    F = 1.0 / (theta_true * (1 - theta_true))    # Fisher info at theta_0
    bvm_var = 1.0 / (F * n)
    bvm_density = stats.norm(theta_mle, np.sqrt(bvm_var)).pdf(theta_grid)

    for (a0, b0), color in zip(priors, ['C0', 'C1', 'C2']):
        post = stats.beta(a0 + k, b0 + n - k).pdf(theta_grid)
        axes[idx].plot(theta_grid, post, color=color, lw=2,
                       label=f'Beta({a0},{b0}) prior')

    axes[idx].plot(theta_grid, bvm_density, 'k--', lw=2,
                   label='BvM Gaussian')
    axes[idx].axvline(theta_true, color='r', ls=':', label='θ₀=0.4')
    axes[idx].set_xlim(0.1, 0.7); axes[idx].set_title(f'n = {n}')
    axes[idx].set_xlabel(r'$\theta$'); axes[idx].grid(alpha=0.3)
    if idx == 0: axes[idx].set_ylabel('posterior density')

axes[0].legend(fontsize=8)
plt.suptitle('BvM 수렴 — prior가 다르더라도 Gaussian으로 모두 수렴')
plt.tight_layout(); plt.savefig('bvm_convergence.png', dpi=150); plt.show()

# ────────────────────────────────────────────────
# TV distance 측정
# ────────────────────────────────────────────────
print(f"{'n':>6} {'TV(Beta(1,1), BvM)':>22} {'TV(Beta(10,10), BvM)':>24}")
for n in [10, 50, 200, 1000, 5000, 20000]:
    data = rng.binomial(1, theta_true, size=n)
    k = data.sum()
    theta_mle = k / n
    bvm_var = theta_true * (1 - theta_true) / n

    for (a0, b0), _ in zip([(1,1), (10,10)], ['', '']):
        post = stats.beta(a0 + k, b0 + n - k).pdf(theta_grid)
        bvm = stats.norm(theta_mle, np.sqrt(bvm_var)).pdf(theta_grid)
        tv = 0.5 * np.trapz(np.abs(post - bvm), theta_grid)
        if (a0, b0) == (1, 1): tv1 = tv
        else: tv2 = tv
    print(f"{n:>6} {tv1:>22.4f} {tv2:>24.4f}")
```

**출력 예시**:
```
     n     TV(Beta(1,1), BvM)  TV(Beta(10,10), BvM)
    10                 0.0895                  0.2134
    50                 0.0382                  0.0891
   200                 0.0189                  0.0367
  1000                 0.0082                  0.0126
  5000                 0.0036                  0.0051
 20000                 0.0018                  0.0024
```

**관찰**: TV 거리 $O(n^{-1/2})$로 감소, prior가 더 informative(Beta(10,10))해도 $n$이 크면 사라짐 — BvM 수치적 확인.

---

## 🔗 AI/ML 연결

### Laplace Approximation의 정당성
BvM이 성립하면 posterior $\approx \mathcal{N}(\hat\theta_{MAP}, H^{-1})$로 **Laplace 근사가 asymptotically exact**. Ch5-02의 이론적 뒷받침.

### BNN에서의 한계
BNN은 고차원·비볼록으로 BvM 엄밀 적용 불가. 그럼에도 "local BvM"으로 **loss landscape의 mode 주변에서는 Gaussian 근사**가 작동 — empirical 관찰.

### Calibration
BvM ⇒ Bayesian posterior가 **automatically frequentist-calibrated** (Ch7-04). ECE가 낮아지는 이론적 이유.

### Bayesian Deep Learning의 근거
"왜 Bayesian DL에서 prior가 중요하지 않아 보이는가?" — 데이터가 많고 네트워크가 크면 (유효) dimension이 제한되어 BvM-like 행동을 보임.

### Hypothesis Testing
BvM → Bayes factor의 asymptotic 거동이 frequentist test와 연결(Schwarz 1978 BIC).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\theta_0$ 존재 | Model misspecification 시 $\theta_0$ 대신 KL-projection $\theta^*$ |
| Identifiability | Mixture, permutation invariant 모델 실패 |
| Smooth likelihood $C^3$ | Non-smooth (ReLU NN, discrete latent) 문제 |
| Fisher $\succ 0$ | Singular Fisher (non-identifiable) → BvM 실패 |
| Interior MLE | Boundary case (variance → 0) 실패 |
| 유한차원 | BNN 등 고차원은 정리 적용 불가 |
| IID 관측 | 시계열·네트워크 데이터 별도 이론 |

**실전 교훈**: BvM은 "대규모 데이터 + 정칙 모델"의 약속. 그 밖에선 prior가 실제로 **중요할 수 있다**.

---

## 📌 핵심 정리

$$\boxed{p(\theta|D_n) \xrightarrow{TV} \mathcal{N}\left(\hat\theta_{MLE}, \frac{F^{-1}(\theta_0)}{n}\right)}$$

4대 귀결:
1. **Prior 잊혀짐** — 정리 5.3
2. **Credible = Confidence** — 정리 5.2
3. **Posterior is Gaussian** — Laplace 근사 정당화
4. **Rate $O(n^{-1/2})$** — 정리 5.4

위반 경우: Non-identifiable / Boundary / Infinite-dim / Non-smooth.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Poisson($\lambda$) + Gamma($a$, $b$) prior에서 posterior Gamma$(a+\sum x_i, b+n)$. $n\to\infty$에서 BvM limit은?

<details>
<summary>해설</summary>

Gamma$(a+n\bar x, b+n)$, $\bar x \to \lambda_0$.

Mean $= (a + n\bar x)/(b + n) \to \lambda_0$.

Variance $= (a + n\bar x)/(b + n)^2 \approx \lambda_0/n$.

Poisson Fisher: $F(\lambda_0) = 1/\lambda_0$, so $F^{-1}/n = \lambda_0/n$. ✓

BvM Gaussian: $\mathcal{N}(\bar x, \lambda_0/n)$. Gamma는 $n$이 크면 점근적으로 Gaussian(Gamma → Gaussian).

</details>

**문제 2** (심화): Mixture model $p(x|\theta) = \pi_1 \mathcal{N}(x; \mu_1, 1) + \pi_2 \mathcal{N}(x; \mu_2, 1)$에서 **label switching** 때문에 BvM이 실패한다. 왜?

<details>
<summary>해설</summary>

$(\mu_1, \mu_2, \pi_1, \pi_2) \leftrightarrow (\mu_2, \mu_1, \pi_2, \pi_1)$이 같은 likelihood를 준다 — **non-identifiable**. Posterior가 **두 개 (또는 $k!$ 개) 대칭 modes**를 가짐.

BvM은 $\theta_0$ 주변 **단일 Gaussian** 수렴을 주장 — multimodal posterior는 이를 위반.

해결책:
- Identifiability 제약 추가 ($\mu_1 < \mu_2$)
- Posterior over "quotient space" (label equivalence class)
- Label-invariant statistics만 reporting

</details>

**문제 3** (AI 연결): BNN에서 BvM이 엄밀히 성립하지 않지만 Laplace approximation이 실전에서 잘 작동한다. 이 역설을 어떻게 이해할 것인가?

<details>
<summary>해설</summary>

BvM의 "유한 $d$, $n \to \infty$" 레짐 대신 BNN은 "$d, n$ 모두 큼" 레짐.

"**Effective dimensionality**" — 실제로 data가 제약하는 차원은 $d$보다 훨씬 작음(over-parameterized network). Loss landscape의 특정 **minimum 주변에선** Gaussian 근사가 작동.

실전: 
- **Post-hoc Laplace** (Ritter 2018) — 학습된 NN의 MAP에서 Hessian 기반 Gaussian
- **KFAC Laplace** — 블록-Kronecker로 고차원 저장
- 단일 mode 내부 curvature만 보고 uncertainty 추정

**한계**: Mode 간 이동(multi-modal)은 못 잡음. Deep ensembles(다른 초기화) 병용으로 보완.

즉 BvM은 "**globally** 엄밀히 성립 안 하지만, **locally** 영감"을 줌.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Predictive Distribution](./04-predictive-distribution.md) | [📚 README로 돌아가기](../README.md) | [Ch2-01. VI의 아이디어와 ELBO 유도 ▶](../ch2-variational-inference/01-vi-idea-elbo.md) |

</div>
