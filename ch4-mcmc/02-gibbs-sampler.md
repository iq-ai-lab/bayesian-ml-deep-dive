# 02. Gibbs Sampler와 조건부 분포

## 🎯 핵심 질문

- 각 차원을 완전조건부 $p(\theta_i|\theta_{-i}, D)$로 업데이트하는 Gibbs가 왜 **MH의 특수경우**인가?
- Gibbs는 왜 **항상 accept** (α = 1)인가?
- Conjugate 구조가 조건부 분포를 어떻게 닫힌형으로 만드는가?
- **Collapsed Gibbs**(일부 latent 주변화)의 효율 이점은?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**LDA**의 표준 추론(Collapsed Gibbs), **Bayesian Mixture**, **Gaussian Mixture posterior**, **HMM**의 Bayesian inference 모두 Gibbs. **Block Gibbs**와 **CAVI**(Ch2-03)의 짝 — 둘 다 conjugate-exp 구조 활용. LDA에서 Collapsed Gibbs가 vanilla Gibbs보다 수백 배 빠름.

---

## 📐 수학적 선행 조건

- [Ch4-01 Metropolis-Hastings](./01-metropolis-hastings.md)
- [Ch1-03 Conjugate priors](../ch1-bayesian-foundation/03-conjugate-priors.md): 조건부 분포의 닫힌형
- [Ch2-03 CAVI](../ch2-variational-inference/03-mean-field-cavi.md): 좌표 업데이트의 variational analog

---

## 📖 직관적 이해

### Gibbs 기본

$\theta = (\theta_1, \ldots, \theta_d)$. 각 차원을 **한 번씩** 완전조건부에서 샘플:

```
Repeat:
   For i = 1, ..., d:
      θ_i ~ p(θ_i | θ_{-i}, D)
```

**완전조건부**(full conditional): 다른 차원 고정, $\theta_i$만의 분포.

### MH와의 관계

Gibbs의 proposal:
$$q(\theta'|\theta) = p(\theta_i'|\theta_{-i}, D) \cdot \mathbf{1}[\theta_{-i}' = \theta_{-i}]$$

이 proposal에서 MH acceptance:
$$\alpha = \min\left(1, \frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)}\right) = 1 \quad \text{(proved below)}$$

**항상 accept** → Gibbs는 **"rejection 없는 MH"**.

### 요리 비유

재료 $\theta_i$를 바꾸는데 **다른 재료 고정 하에서 "최적 분포"에서 샘플**. 매번 수락 — 리젝션 낭비 없음.

---

## ✏️ 엄밀한 정의

### 정의 2.1 — Gibbs Sampler

$\theta = (\theta_1, \ldots, \theta_d)$, posterior $\pi$. **Systematic-scan Gibbs**:

```
while not converged:
    for i = 1 to d:
        θ_i^{(t+1)} ~ p(θ_i | θ_1^{(t+1)}, ..., θ_{i-1}^{(t+1)}, θ_{i+1}^{(t)}, ..., θ_d^{(t)}, D)
```

**Random-scan**: 매 step에서 $i$를 랜덤 선택.

### 정의 2.2 — Full Conditional

$$p(\theta_i | \theta_{-i}, D) = \frac{p(\theta_i, \theta_{-i}, D)}{p(\theta_{-i}, D)} \propto p(\theta_i, \theta_{-i}, D)$$

$\theta_i$에 관한 부분만 남음.

### 정의 2.3 — Block Gibbs

$\theta$를 **block**들로 나눔: $\{\theta_{B_1}, \ldots, \theta_{B_K}\}$. 각 block 전체를 조건부에서 한 번에 샘플:

$$\theta_{B_k}^{(t+1)} \sim p(\theta_{B_k} | \theta_{-B_k}, D)$$

Correlated block은 **함께** 움직여 mixing 개선.

### 정의 2.4 — Collapsed Gibbs

일부 latent $\theta_A$를 **해석적으로 주변화**:
$$p(\theta_B | D) = \int p(\theta_A, \theta_B | D)d\theta_A$$

축소된 공간에서 Gibbs → 차원 감소 + mixing 개선.

---

## 🔬 정리와 증명

### 정리 2.1 — Gibbs는 MH의 특수경우 with α = 1

**명제**: Gibbs proposal $q(\theta'|\theta) = p(\theta_i'|\theta_{-i}, D)\mathbf{1}[\theta_{-i}' = \theta_{-i}]$에서 MH acceptance:

$$\alpha(\theta, \theta') = 1$$

**증명**: 

$\theta_{-i} = \theta_{-i}'$일 때 (Gibbs는 한 차원만 움직임):

$$\frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)} = \frac{p(\theta_i', \theta_{-i}|D) \cdot p(\theta_i|\theta_{-i}, D)}{p(\theta_i, \theta_{-i}|D) \cdot p(\theta_i'|\theta_{-i}, D)}$$

$p(\theta_i, \theta_{-i}|D) = p(\theta_i|\theta_{-i}, D)p(\theta_{-i}|D)$ (Bayes):

$$= \frac{p(\theta_i'|\theta_{-i}, D)p(\theta_{-i}|D) \cdot p(\theta_i|\theta_{-i}, D)}{p(\theta_i|\theta_{-i}, D)p(\theta_{-i}|D) \cdot p(\theta_i'|\theta_{-i}, D)} = 1$$

$\alpha = \min(1, 1) = 1$. $\square$

**귀결**: Gibbs 샘플은 항상 수락 → **평균 step당 유의미한 움직임**. MH의 rejection 비용 없음.

### 정리 2.2 — Gibbs의 Detailed Balance

**명제**: Gibbs transition $T$는 정상분포 $\pi$를 갖는다.

**증명**: 정리 2.1 + Ch4-01 정리 1.1, 1.2. Gibbs가 MH의 case이므로 detailed balance 자동 성립, 따라서 $\pi$가 정상. $\square$

### 정리 2.3 — 2D Gaussian에서 Gibbs의 명시적 형태

**명제**: $\theta = (\theta_1, \theta_2) \sim \mathcal{N}(0, \Sigma)$, $\Sigma = \begin{pmatrix}1 & \rho\\ \rho & 1\end{pmatrix}$. 완전조건부:

$$\theta_1 | \theta_2 \sim \mathcal{N}(\rho\theta_2, 1 - \rho^2)$$
$$\theta_2 | \theta_1 \sim \mathcal{N}(\rho\theta_1, 1 - \rho^2)$$

**증명**: Joint Gaussian conditional formula. $\square$

**관찰**: $|\rho| \to 1$에서 Gibbs가 **강한 correlation을 천천히 탐색** → mixing 느림. Block Gibbs(2D whole)로 해결 가능.

### 정리 2.4 — Collapsed Gibbs의 분산 감소

**명제** (Liu 1994): $\theta_A$를 주변화하면 Gibbs의 **asymptotic variance가 감소**:

$$\text{Var}_{\text{collapsed}}[\hat f] \leq \text{Var}_{\text{full}}[\hat f]$$

**증명 스케치**: Rao-Blackwellization:
$$\text{Var}[f(\theta_B)] \leq \text{Var}[f(\theta_A, \theta_B)]$$
for $\theta_A$-marginalized estimators. Gibbs도 같은 원리. $\square$

**실전 영향**: LDA에서 **word-topic assignment** $z$를 주변화 → **$\theta$ (topic distribution)과 $\phi$ (word distribution)만 Gibbs**. 수십 배 빠른 수렴.

### 정리 2.5 — Gibbs의 Mixing과 조건부 Correlation

**명제** (Amit 1991, 2D Gaussian): 강한 correlation $|\rho| \to 1$에서 Gibbs의 relaxation rate:

$$\text{rate} \approx 1 - \rho^2$$

$\rho = 0.99$이면 rate $\approx 0.02$ → 1/50 mixing. Block Gibbs가 이 문제를 즉시 해결.

### 예시 — Bayesian Linear Regression의 Gibbs

$y = X\beta + \epsilon, \epsilon \sim \mathcal{N}(0, \sigma^2 I)$. Priors: $\beta \sim \mathcal{N}(0, \tau^2 I), 1/\sigma^2 \sim \text{Gamma}$.

완전조건부:
- $\beta | \sigma^2, D \sim \mathcal{N}((X^TX/\sigma^2 + I/\tau^2)^{-1}X^Ty/\sigma^2, (X^TX/\sigma^2 + I/\tau^2)^{-1})$
- $\sigma^2 | \beta, D \sim \text{Inv-Gamma}$

Gibbs가 closed form을 반복 활용.

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Gibbs for 2D correlated Gaussian
# π(θ₁, θ₂) = N(0, [[1, ρ], [ρ, 1]])
# ────────────────────────────────────────────────
rho = 0.9
n_iter = 10000

theta = np.zeros((n_iter, 2))
theta[0] = [3.0, 3.0]  # init far from mean

for t in range(1, n_iter):
    # θ₁ | θ₂ ~ N(ρθ₂, 1-ρ²)
    theta[t, 0] = rho*theta[t-1, 1] + np.sqrt(1-rho**2)*rng.standard_normal()
    # θ₂ | θ₁ ~ N(ρθ₁, 1-ρ²)
    theta[t, 1] = rho*theta[t, 0] + np.sqrt(1-rho**2)*rng.standard_normal()

burn = 1000
samples = theta[burn:]
print(f"Empirical mean: {samples.mean(axis=0)}")
print(f"Empirical cov: \n{np.cov(samples.T)}")
print(f"Expected: mean=[0,0], cov=[[1,{rho}],[{rho},1]]")

# ────────────────────────────────────────────────
# Compare Gibbs (coordinate) vs Block (joint)
# ────────────────────────────────────────────────
# Block Gibbs: sample (θ₁, θ₂) jointly from true Gaussian each iter (for this toy, trivial)
block_samples = rng.multivariate_normal([0, 0], [[1, rho], [rho, 1]], size=n_iter)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Trace plot first 500
axes[0, 0].plot(theta[:500, 0], lw=0.7, label=r'$\theta_1$')
axes[0, 0].plot(theta[:500, 1], lw=0.7, label=r'$\theta_2$')
axes[0, 0].set_title(f'Gibbs trace (ρ={rho}) — zigzag pattern (slow mixing)')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# 2D scatter
axes[0, 1].scatter(samples[:500, 0], samples[:500, 1], s=5, alpha=0.5, label='Gibbs path')
axes[0, 1].set_title('Gibbs samples (first 500, showing correlation)')
axes[0, 1].grid(alpha=0.3)

# Autocorrelation
from statsmodels.tsa.stattools import acf
for i, (data, label) in enumerate([(samples[:, 0], 'Gibbs θ₁'),
                                     (block_samples[:, 0], 'Block (iid)')]):
    ax = axes[1, i]
    ax.stem(acf(data, nlags=40), basefmt=" ")
    ax.set_title(f'Autocorrelation: {label}')
    ax.set_xlabel('lag'); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('gibbs_demo.png', dpi=150); plt.show()

# ESS-like: how many iid samples equivalent
print(f"Gibbs: ~{n_iter/(1 + 2*np.sum(acf(samples[:, 0], nlags=50)[1:])):.0f} effective samples")
```

---

## 🔗 AI/ML 연결

### LDA (Latent Dirichlet Allocation)
Collapsed Gibbs가 표준 알고리즘 (Griffiths & Steyvers 2004). 문서별 topic, topic별 word assignment를 해석적으로 collapse.

### Bayesian GMM
$\mu_k, \pi_k, z_i$ Gibbs. Ch2-03의 CAVI와 쌍.

### HMM Bayesian
Forward-backward + Gibbs를 조합 (forward filtering, backward sampling).

### Probabilistic Programming
Stan은 NUTS 위주 (continuous 전용) — discrete parameter 있으면 PyMC의 `Categorical`이 MH with Gibbs로 대체.

### Ising Model
각 spin flip = 1D Gibbs step. Statistical physics와 교차.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| 완전조건부 샘플 가능 | 비공액 시 MH-within-Gibbs 사용 |
| Conjugate-exp 구조 | 없으면 각 조건부도 intractable |
| Low correlation between dims | 강한 correlation → slow mixing (정리 2.5) |
| Discrete 잘 다룸 | Continuous에선 HMC가 더 효율적 |

**실무 팁**: 
- **Block Gibbs**: correlated dimensions 묶기
- **Collapsed Gibbs**: 해석 주변화 가능한 latent는 integrate out
- **MH-within-Gibbs**: conditional이 닫힌형 아닐 때

---

## 📌 핵심 정리

$$\boxed{\theta_i^{(t+1)} \sim p(\theta_i | \theta_{-i}^{(t)}, D)}$$

핵심:
- Gibbs = **항상 accept**하는 MH
- Conjugate ⇒ 조건부 닫힌형
- Collapsed Gibbs로 차원 감소 + 분산 감소
- 강한 correlation에서 slow mixing → Block Gibbs

---

## 🤔 생각해볼 문제

**문제 1** (기초): Beta-Bernoulli posterior Beta(9, 5)에 대한 "Gibbs"는? (1D이므로 단순화)

<details>
<summary>해설</summary>

1D이므로 **완전조건부 = posterior itself** = Beta(9, 5). 직접 `scipy.stats.beta(9, 5).rvs()`로 샘플.

Gibbs가 "non-trivial"하려면 **multivariate** — 다차원에서 각 차원별 조건부가 단순해야 효용.

예시: Bayesian mixed model의 $(\mu, \sigma^2, z_i)$ 각자 conditional이 Gaussian, Inv-Gamma, Categorical.

</details>

**문제 2** (심화): LDA의 collapsed Gibbs에서 무엇을 주변화하고 왜?

<details>
<summary>해설</summary>

LDA: document $d$에서 word $w$, topic assignment $z$.
- $\theta_d$ = document's topic dist (Dir prior)
- $\phi_k$ = topic's word dist (Dir prior)
- $z_{dw}$ = assigned topic

**Collapsed**: $\theta, \phi$ 주변화 (Dirichlet conjugate로 해석):

$$p(z_{dw} = k | z_{-dw}, w) \propto \frac{n_{dk}^{-dw} + \alpha}{n_{d\cdot}^{-dw} + K\alpha} \cdot \frac{n_{kw}^{-dw} + \beta}{n_{k\cdot}^{-dw} + V\beta}$$

(count-based, simple update).

**이점**:
- Dimensionality: $\{\theta, \phi, z\} \to \{z\}$만 샘플 (parameter count 대폭 감소)
- Mixing: $z$와 $(\theta, \phi)$의 strong coupling 제거 → 빠른 수렴
- Rao-Blackwellization: 분산 감소 (정리 2.4)

결과: 수백 배 속도 향상. LDA 실전 추론의 표준.

</details>

**문제 3** (AI 연결): Gibbs와 **CAVI**(Ch2-03)의 구조적 유사성은?

<details>
<summary>해설</summary>

둘 다 **같은 조건부 구조** 활용:

Gibbs: $\theta_i \sim p(\theta_i | \theta_{-i}, D)$ (샘플)
CAVI: $q_i \propto \exp(\mathbb{E}_{q_{-i}}[\log p(\theta, D)])$ (변분 근사)

조건부 분포가 exp family면:
- Gibbs: 그 family에서 sample
- CAVI: 그 family의 natural param을 기댓값으로

같은 수학 구조, 다른 semantics:
- Gibbs → exact samples, stochastic
- CAVI → deterministic approximation

**Hybrid**: SVGD, sequential VI — 양쪽 장점 통합.

"**Conjugate-exp 구조 활용**"이 두 알고리즘의 공통 원리 — 이것을 모르면 둘 다 이해 불가.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Metropolis-Hastings](./01-metropolis-hastings.md) | [📚 README로 돌아가기](../README.md) | [03. Hamiltonian Monte Carlo (HMC) ▶](./03-hamiltonian-monte-carlo.md) |

</div>
