# 01. Metropolis-Hastings 재정리

## 🎯 핵심 질문

- Proposal $q(\theta'|\theta)$ + acceptance $\alpha = \min(1, \frac{p(\theta'|D)q(\theta|\theta')}{p(\theta|D)q(\theta'|\theta)})$이 왜 **정확한 posterior 샘플**을 주는가?
- **Detailed balance** $\pi(\theta)T(\theta\to\theta') = \pi(\theta')T(\theta'\to\theta)$가 어떻게 증명되는가?
- Evidence $p(D)$ 없이도 MCMC가 posterior에 수렴하는 이유는?
- Random-walk MH의 **acceptance rate** 최적값은 왜 ≈ 0.234 (고차원에서)?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**PyMC, Stan, NumPyro** 내부의 기본 MCMC가 MH. Evidence intractable이더라도 **posterior 샘플 가능**이 핵심. **BNN의 HMC**(Ch4-03), **Gibbs의 block updates**(Ch4-02)는 MH의 특수경우. **Reversible jump MCMC**, **MH within Gibbs** 같은 고급 기법의 수학적 토대.

---

## 📐 수학적 선행 조건

- [Ch1-01 Bayes 정리](../ch1-bayesian-foundation/01-bayes-rule-four-roles.md): posterior ∝ likelihood × prior
- [Stochastic Processes Deep Dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive): Markov chain, ergodicity, stationary distribution
- Detailed balance 개념

---

## 📖 직관적 이해

### 기본 아이디어

정확한 posterior $\pi(\theta) = p(\theta|D)$에서 샘플하고 싶지만 직접 불가능. 대신 **Markov chain** $\theta_0, \theta_1, \theta_2, \ldots$를 구성하여 **정상분포가 $\pi$**가 되게 하면:

$$\frac{1}{T}\sum_{t=1}^T f(\theta_t) \to \mathbb{E}_\pi[f]$$

(ergodic theorem). 충분히 오래 돌리면 **샘플의 히스토그램이 posterior**.

### MH 알고리즘

```
Given current θ:
1. Propose θ' ~ q(θ' | θ)
2. Compute α = min(1, π(θ') q(θ|θ') / (π(θ) q(θ'|θ)))
3. With probability α: θ_next = θ'  (accept)
   Else: θ_next = θ                 (reject)
```

**핵심**: $\pi(\theta')/\pi(\theta)$만 필요 → **상수 $p(D)$ 상쇄** → evidence 몰라도 됨.

### Detailed Balance 의미

"정방향 flow = 역방향 flow":
$$\pi(\theta)T(\theta\to\theta') = \pi(\theta')T(\theta'\to\theta)$$

모든 $(\theta, \theta')$ 쌍에 대해. 이 **국소 균형**이 **전역 정상분포**를 보장.

### 요리 비유

- 현재 메뉴 $\theta$
- 제안 $\theta'$ (한 재료 바꿈)
- "새 메뉴의 점수 / 현재 점수"로 확률적 accept
- 긴 시간 돌리면 **점수 비례로 메뉴를 방문** → posterior 샘플

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Transition Kernel

$$T(\theta\to\theta') = q(\theta'|\theta)\alpha(\theta, \theta') + \delta_\theta(\theta')\left(1 - \int q(\theta''|\theta)\alpha(\theta, \theta'')d\theta''\right)$$

- 첫 항: 제안 $\theta'$ 수락
- 둘째 항: 거부 → 현재 $\theta$에 머뭄

### 정의 1.2 — MH Acceptance

$$\alpha(\theta, \theta') = \min\left(1, \frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)}\right)$$

### 정의 1.3 — Detailed Balance

$$\pi(\theta)T(\theta\to\theta') = \pi(\theta')T(\theta'\to\theta) \quad \forall\theta, \theta'$$

### 정의 1.4 — Symmetric Proposal (Random-Walk MH)

$q(\theta'|\theta) = q(\theta|\theta')$ (예: $\theta' = \theta + \mathcal{N}(0, \Sigma)$). 이 경우:

$$\alpha = \min(1, \pi(\theta')/\pi(\theta))$$

**Metropolis 원 알고리즘** (1953).

---

## 🔬 정리와 증명

### 정리 1.1 — MH는 Detailed Balance 만족

**명제**: $\alpha(\theta, \theta') = \min(1, R)$, $R = \pi(\theta')q(\theta|\theta')/[\pi(\theta)q(\theta'|\theta)]$로 정의하면:

$$\pi(\theta)q(\theta'|\theta)\alpha(\theta, \theta') = \pi(\theta')q(\theta|\theta')\alpha(\theta', \theta)$$

**증명**:

Case 1: $R \leq 1$ (즉 $\alpha(\theta, \theta') = R$)
- $1/R = \pi(\theta)q(\theta'|\theta)/[\pi(\theta')q(\theta|\theta')] \geq 1$
- $\alpha(\theta', \theta) = \min(1, 1/R) = 1$

좌변:
$$\pi(\theta)q(\theta'|\theta) R = \pi(\theta)q(\theta'|\theta) \cdot \frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)} = \pi(\theta')q(\theta|\theta')$$

우변:
$$\pi(\theta')q(\theta|\theta')\cdot 1 = \pi(\theta')q(\theta|\theta')$$

동일 ✓.

Case 2: $R > 1$ — 대칭. $\square$

### 정리 1.2 — Detailed Balance ⇒ $\pi$가 정상분포

**명제**: $\pi T = \pi$ (즉 $\pi$가 정상).

**증명**: Detailed balance를 $\theta$로 적분:

$$\int \pi(\theta)T(\theta\to\theta')d\theta = \int \pi(\theta')T(\theta'\to\theta)d\theta = \pi(\theta')\int T(\theta'\to\theta)d\theta = \pi(\theta')$$

(Transition kernel이 total probability 1). $\square$

### 정리 1.3 — Evidence의 상쇄

**명제**: $\pi(\theta) = p(D|\theta)p(\theta)/p(D)$. Acceptance ratio:

$$\frac{\pi(\theta')}{\pi(\theta)} = \frac{p(D|\theta')p(\theta')/p(D)}{p(D|\theta)p(\theta)/p(D)} = \frac{p(D|\theta')p(\theta')}{p(D|\theta)p(\theta)}$$

**$p(D)$가 자동 상쇄** — intractable evidence 없이도 계산 가능. $\square$

**이것이 MCMC의 마법**: Bayesian의 근본 문제(evidence)를 우회.

### 정리 1.4 — Ergodicity (Convergence)

**명제**: Transition $T$가:
1. **Irreducible** (어떤 두 state 사이에도 양의 prob path)
2. **Aperiodic** (주기 없음)

이면, 임의 initial $\theta_0$에 대해:

$$T^n(\theta_0, \cdot) \xrightarrow{TV} \pi \quad (n \to \infty)$$

**증명**: Markov chain ergodic theorem (Stochastic Processes 레포 참조). $\square$

**귀결**: 충분한 burn-in 후 샘플이 $\pi$에서 나온 것처럼 사용 가능.

### 정리 1.5 — Random-Walk MH의 최적 Acceptance Rate

**명제** (Roberts, Gelman, Gilks 1997): $d$차원 Gaussian target $\pi$, random-walk $\mathcal{N}(\theta, \Sigma)$ proposal에서 ESS를 최대화하는 acceptance rate:

$$d = 1: \alpha^* \approx 0.44$$
$$d \to \infty: \alpha^* \approx 0.234$$

**증명 스케치**: Scaling $\Sigma = (\ell/\sqrt d)^2 \Sigma_\pi$에서 chain의 diffusion limit 분석. Optimal scale $\ell^* \approx 2.4$, $\alpha^* \to 0.234$. $\square$

**실전 가이드**:
- Acceptance rate 너무 높음 (>0.7): proposal step이 너무 작음 → chain slow mixing
- 너무 낮음 (<0.1): step이 너무 큼 → rejections 많음
- **Adaptive MH** (Haario 2001): 자동 튜닝.

### 예시 — 1D Gaussian posterior 샘플링

$\pi(\theta) = \mathcal{N}(0, 1)$. Proposal $\theta' = \theta + 2\epsilon, \epsilon\sim\mathcal{N}(0,1)$ (symmetric).

Acceptance: $\alpha = \min(1, \exp(-(\theta'^2 - \theta^2)/2))$.

$T = 10000$ steps, burn-in 1000 → 샘플 히스토그램이 $\mathcal{N}(0,1)$.

---

## 💻 NumPy 구현 검증

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# MH for 1D Beta posterior
# π(θ) ∝ θ^(α-1)(1-θ)^(β-1), Beta(9, 5) (= Beta-Bernoulli posterior)
# ────────────────────────────────────────────────
alpha_p, beta_p = 9.0, 5.0  # Beta(9,5) posterior params

def log_pi(theta):
    if theta <= 0 or theta >= 1: return -np.inf
    return (alpha_p-1)*np.log(theta) + (beta_p-1)*np.log(1-theta)

# Random-walk MH
def mh_sample(n_iter, step, theta0=0.5):
    samples = np.zeros(n_iter)
    theta = theta0
    n_accept = 0
    for t in range(n_iter):
        theta_prop = theta + step * rng.standard_normal()
        log_r = log_pi(theta_prop) - log_pi(theta)
        if np.log(rng.random()) < log_r:
            theta = theta_prop
            n_accept += 1
        samples[t] = theta
    return samples, n_accept / n_iter

# 세 step size 비교
fig, axes = plt.subplots(3, 2, figsize=(12, 10))
theta_grid = np.linspace(0.01, 0.99, 300)
true_post = stats.beta(alpha_p, beta_p).pdf(theta_grid)

for i, step in enumerate([0.01, 0.2, 3.0]):
    samples, ar = mh_sample(20000, step)
    burn = samples[5000:]
    axes[i, 0].plot(samples[:2000], lw=0.5)
    axes[i, 0].set_title(f'Trace (step={step}, accept rate={ar:.2f})')
    axes[i, 0].set_ylabel(r'$\theta$')
    
    axes[i, 1].hist(burn, bins=50, density=True, alpha=0.5, label='MH samples')
    axes[i, 1].plot(theta_grid, true_post, 'r-', lw=2, label='True Beta(9,5)')
    axes[i, 1].set_title(f'Posterior approximation (step={step})')
    axes[i, 1].legend(); axes[i, 1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig('mh_demo.png', dpi=150); plt.show()

# ────────────────────────────────────────────────
# Detailed balance 수치 검증
# ────────────────────────────────────────────────
print("\nDetailed balance check (MH is EXACT up to finite-sample noise):")
theta1, theta2 = 0.4, 0.7
step = 0.3
# LHS: π(θ1) · q(θ2|θ1) · α(θ1, θ2)
q12 = stats.norm(theta1, step).pdf(theta2)
q21 = stats.norm(theta2, step).pdf(theta1)
pi1 = np.exp(log_pi(theta1)); pi2 = np.exp(log_pi(theta2))
r_12 = pi2 * q21 / (pi1 * q12)
alpha_12 = min(1, r_12)
alpha_21 = min(1, 1/r_12)
LHS = pi1 * q12 * alpha_12
RHS = pi2 * q21 * alpha_21
print(f"LHS = π(θ1)q(θ2|θ1)α(θ1,θ2) = {LHS:.6f}")
print(f"RHS = π(θ2)q(θ1|θ2)α(θ2,θ1) = {RHS:.6f}")
print(f"Match: {np.isclose(LHS, RHS)}")
```

---

## 🔗 AI/ML 연결

### Gibbs Sampler (Ch4-02)
각 차원 MH with proposal = 완전조건부. Always accept.

### HMC (Ch4-03)
Gradient-aware MH. Proposal은 Leapfrog integration 결과, acceptance로 discretization 오차 수정.

### NUTS (Ch4-04)
HMC의 trajectory length 자동 결정.

### Bayesian Model Comparison
$p(M_k|D) \propto p(D|M_k) p(M_k)$ 계산. Reversible jump MCMC가 모델 간 이동.

### PyMC/Stan
내부적으로 NUTS(HMC variant) 사용, fallback to Metropolis for discrete.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Proposal symmetric or tractable | Optimal proposal 설계 어려움 |
| Posterior이 single-mode or connected | **Multimodal**에서 mode 간 이동 drasticly 느림 |
| Low dimension | 고차원에서 random-walk MH는 지수적으로 느림 → HMC 필요 |
| Acceptance rate 튜닝 가능 | 자동화 없이 hand-tune 필요 (Adaptive MH 일부 해결) |

**실무 팁**: 
- 2D~10D: MH 충분
- 10D~100D: HMC
- >100D (BNN): HMC + data subsampling, SGLD

---

## 📌 핵심 정리

$$\boxed{\alpha = \min\left(1, \frac{\pi(\theta')q(\theta|\theta')}{\pi(\theta)q(\theta'|\theta)}\right)}$$

핵심:
- **Detailed balance** ⇒ $\pi$가 정상분포
- Evidence $p(D)$ **자동 상쇄**
- Ergodicity로 convergence 보장
- Random-walk: 고차원에서 $\alpha^* \approx 0.234$

---

## 🤔 생각해볼 문제

**문제 1** (기초): Proposal $q$가 **대칭**일 때 acceptance는 어떻게 단순화?

<details>
<summary>해설</summary>

$q(\theta'|\theta) = q(\theta|\theta')$ → 비율 $q(\theta|\theta')/q(\theta'|\theta) = 1$.

$\alpha = \min(1, \pi(\theta')/\pi(\theta))$.

이것이 **Metropolis (원)** 알고리즘. $\pi$가 **potential energy의 Boltzmann**이면 $\alpha = \min(1, e^{-\Delta U/T})$ — physics의 Metropolis step.

비대칭 proposal(e.g., Langevin-based)은 $q$ 비율 필수.

</details>

**문제 2** (심화): Burn-in이 끝났는지 어떻게 알 수 있나? $\hat R$ (Ch4-05) 외에 다른 진단?

<details>
<summary>해설</summary>

- **Trace plot**: 육안으로 mixing 관찰, "caterpillar" pattern 원함
- **Autocorrelation**: 급격히 감소해야
- **Running mean**: 안정
- **Multiple chains from dispersed inits**: same posterior로 수렴
- **ESS** (Effective Sample Size, Ch4-05)
- **$\hat R$** (Gelman-Rubin): 여러 체인의 within/between variance

실전 체크리스트:
1. $\hat R < 1.01$
2. ESS > 400 for every parameter
3. Trace plots 육안 OK
4. No divergences (HMC)

</details>

**문제 3** (AI 연결): BNN posterior가 $10^6$차원인데 random-walk MH는 왜 작동 불가?

<details>
<summary>해설</summary>

- Proposal scale $\sigma$를 너무 작게 → 1 step에 거의 안 움직임, $10^6$개 parameter 독립 탐색 불가
- 너무 크게 → 거의 항상 reject
- 이론: optimal $\sigma \sim O(1/\sqrt d)$ → exploration rate $O(1/d)$ → mixing time $O(d)$ or worse

**$10^6$ dimension**: 실용적으로 수렴 불가능. HMC는 이 문제를 gradient 활용으로 해결 — **dimension-independent mixing** (idealized).

심지어 HMC도 BNN엔 느림 → **SGLD, SWAG, MC Dropout** 같은 approximation 사용(Ch5-04, 05).

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch3-05 IWAE](../ch3-vae-modern-variational/05-iwae.md) | [📚 README로 돌아가기](../README.md) | [02. Gibbs Sampler와 조건부 분포 ▶](./02-gibbs-sampler.md) |

</div>
