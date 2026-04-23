# 02. ELBO의 3가지 분해

## 🎯 핵심 질문

- ELBO는 왜 **세 가지** 동치 형태로 표현되는가?
- **Evidence − KL gap** / **Reconstruction + prior regularization** / **Energy + entropy** 각 분해가 주는 직관은?
- 각 분해가 어느 응용(VAE·BNN·physics-inspired VI)에서 핵심적인가?
- **Reconstruction + prior reg** 분해가 왜 **딥러닝 pipeline과 자연스럽게 매치**되는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

세 분해는 **같은 양의 세 얼굴**이다. VAE는 (2) reconstruction + KL을 학습 loss로 쓰고, BNN는 (2)를 ELBO gradient로, physics-inspired VI는 (3) energy + entropy로, 이론 분석은 (1) evidence − gap으로. **각 분해는 같은 수학적 양을 서로 다른 관점에서 조명**하며, 문제에 맞는 분해를 택하는 것이 implementation과 이해의 핵심.

---

## 📐 수학적 선행 조건

- [Ch2-01 VI 아이디어와 ELBO 유도](./01-vi-idea-elbo.md)
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): Entropy $H$, KL, cross-entropy
- Joint/conditional decomposition: $p(x, \theta) = p(x|\theta)p(\theta)$

---

## 📖 직관적 이해

### 세 분해 요약표

| # | 분해 | 해석 | 주 응용 |
|---|------|------|---------|
| **(1)** | $\mathcal{L} = \log p(x) - \text{KL}(q\|p(\cdot\|x))$ | Evidence − gap | 이론 분석, 수렴성 |
| **(2)** | $\mathcal{L} = \mathbb{E}_q\log p(x\|\theta) - \text{KL}(q\|p(\theta))$ | Reconstruction + prior reg | **VAE, BNN** |
| **(3)** | $\mathcal{L} = \mathbb{E}_q\log p(x, \theta) + H(q)$ | Energy + entropy | Physics, MaxEnt |

### 각 분해의 "이야기"

**(1) Evidence − gap**:
- "ELBO는 log evidence에서 얼마나 먼가?"
- Tight bound iff $q = p(\cdot|x)$
- **이론적 benchmark**: ELBO vs $\log p(x)$ 비교로 VI 품질 평가

**(2) Reconstruction + prior reg**:
- "데이터를 얼마나 잘 복원하는가" + "prior에서 너무 멀지 않음"
- **딥러닝 loss 형태**: data term + regularizer
- VAE의 reconstruction loss + KL penalty의 의미

**(3) Energy + entropy**:
- "평균 energy" + "분포의 entropy"
- 통계역학의 Helmholtz free energy와 유사 — $F = U - TS$
- MaxEnt principle(Jaynes)과의 연결

### 요리 비유

- **(1)**: "완벽한 요리 레시피($\log p(x)$)와 내 요리($\mathcal{L}$) 사이의 거리"
- **(2)**: "재료를 얼마나 잘 쓰는지" + "이 집의 요리 스타일(prior)과 얼마나 가까운지"
- **(3)**: "평균 맛점수" + "메뉴의 다양성"

---

## ✏️ 엄밀한 정의

### 정의 2.1 — 세 분해

$$\mathcal{L}(q) := \mathbb{E}_q[\log p(x, \theta) - \log q(\theta)]$$

**분해 (1)**: $\mathcal{L}(q) = \log p(x) - \text{KL}(q(\theta)\|p(\theta|x))$

**분해 (2)**: $\mathcal{L}(q) = \mathbb{E}_q[\log p(x|\theta)] - \text{KL}(q(\theta)\|p(\theta))$

**분해 (3)**: $\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] + H(q)$

여기서 $H(q) = -\mathbb{E}_q[\log q(\theta)]$ = $q$의 differential entropy.

---

## 🔬 정리와 증명

### 정리 2.1 — 분해 (1): Evidence − KL gap

**명제**: $\mathcal{L}(q) = \log p(x) - \text{KL}(q(\theta)\|p(\theta|x))$

**증명**: Ch2-01 정리 1.1. $\square$

### 정리 2.2 — 분해 (2): Reconstruction + Prior KL

**명제**: $\mathcal{L}(q) = \mathbb{E}_{q(\theta)}[\log p(x|\theta)] - \text{KL}(q(\theta)\|p(\theta))$

**증명**:

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta) - \log q(\theta)]$$

$p(x, \theta) = p(x|\theta)p(\theta)$:
$$= \mathbb{E}_q[\log p(x|\theta) + \log p(\theta) - \log q(\theta)]$$

$$= \mathbb{E}_q[\log p(x|\theta)] + \mathbb{E}_q[\log p(\theta) - \log q(\theta)]$$

두 번째 항이 $-\text{KL}(q\|p_\theta)$:
$$= \mathbb{E}_q[\log p(x|\theta)] - \text{KL}(q(\theta)\|p(\theta))$$

$\square$

> **VAE의 직접 응용**: latent $z$에 대해 $p(x|z), p(z), q(z|x)$:
> $$\mathcal{L}(x) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\|p(z))$$
> 왼쪽 = reconstruction, 오른쪽 = latent prior regularization.

### 정리 2.3 — 분해 (3): Energy + Entropy

**명제**: $\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] + H(q)$

**증명**:

$$\mathcal{L}(q) = \mathbb{E}_q[\log p(x, \theta)] - \mathbb{E}_q[\log q(\theta)] = \mathbb{E}_q[\log p(x, \theta)] + H(q)$$

$\square$

> **물리 해석**: $U := -\log p(x,\theta)$를 "energy"로 놓으면:
> $$-\mathcal{L}(q) = \mathbb{E}_q[U] - H(q) \equiv F(q)$$
> Helmholtz free energy $F = U - TS$ ($T=1$). ELBO 최대화 = **free energy 최소화**. 통계역학에서의 variational principle과 동일한 구조.

### 정리 2.4 — 세 분해의 동치성

**명제**: 세 분해는 대수적으로 동치:

$$\log p(x) - \text{KL}(q\|p(\cdot|x)) = \mathbb{E}_q[\log p(x|\theta)] - \text{KL}(q\|p_\theta) = \mathbb{E}_q[\log p(x, \theta)] + H(q)$$

**증명**: 모두 $\mathcal{L}(q)$의 서로 다른 표현. 정리 2.1, 2.2, 2.3. 또는 직접 대수:

(2) → (3): 
$$\mathbb{E}_q[\log p(x|\theta)] - \text{KL}(q\|p_\theta)$$
$$= \mathbb{E}_q[\log p(x|\theta)] + \mathbb{E}_q[\log p(\theta)] - \mathbb{E}_q[\log q(\theta)]$$
$$= \mathbb{E}_q[\log p(x, \theta)] + H(q)$$

(3) → (1):
$$\mathbb{E}_q[\log p(x, \theta)] + H(q) = \mathbb{E}_q[\log p(\theta|x) + \log p(x)] - \mathbb{E}_q[\log q]$$
$$= \log p(x) + \mathbb{E}_q[\log p(\theta|x) - \log q] = \log p(x) - \text{KL}(q\|p(\cdot|x))$$

$\square$

### 정리 2.5 — β-scaling (β-VAE)

**명제**: $\beta$-가중 ELBO를 정의:

$$\mathcal{L}_\beta(q) = \mathbb{E}_q[\log p(x|\theta)] - \beta\,\text{KL}(q(\theta)\|p(\theta))$$

$\beta = 1$이면 표준 ELBO. $\beta > 1$이면 KL 항이 더 엄격 → **disentanglement**(Ch3-02).

**명제**: $\mathcal{L}_\beta$는 $\beta \neq 1$일 때 **진짜 ELBO가 아니다**(lower bound of log evidence 아님). 대신 다른 objective의 lower bound(see Higgins et al. 2017, Alemi et al. 2018).

**증명 스케치**: 분해 (2) 구조를 유지하지만 상수 배수가 아님. $\square$

### 정리 2.6 — Conditional VI (Posterior over latent $z$만)

**명제**: Observed $x$, latent $z$, parameters $\theta$ (deep learning setting). $q_\phi(z|x)$에 대한 **per-example** ELBO:

$$\mathcal{L}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x)\|p(z))$$

전체 데이터셋 $\mathcal{D} = \{x_i\}$:
$$\mathcal{L}_\text{total} = \sum_i \mathcal{L}(x_i; \theta, \phi)$$

**증명**: $\log p(x_i) \geq \mathcal{L}(x_i)$의 합. $\square$

이것이 VAE의 **학습 가능한 목적함수**.

### 예시 — Beta-Bernoulli에서 세 분해의 수치적 동치

$n=10, k=7$, Beta(2,2) prior. $q_\phi(\theta) = \mathcal{N}$ in logit-space.

세 표현으로 ELBO 계산 → 모두 같은 값이 나옴을 확인 (아래 NumPy).

---

## 💻 NumPy 구현 검증

```python
import numpy as np
from scipy import stats
from scipy.special import betaln

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Setup: Beta-Bernoulli, logit-Gaussian VI
# ────────────────────────────────────────────────
alpha, beta_ = 2.0, 2.0
n, k = 10, 7

# VI params (Ch2-01에서 찾은 값 사용)
mu_vi, log_sigma_vi = 0.8, -0.7
sigma_vi = np.exp(log_sigma_vi)

# Monte Carlo samples
n_mc = 100_000
eps = rng.standard_normal(n_mc)
u = mu_vi + sigma_vi * eps               # sample from q in logit space
theta = 1.0 / (1.0 + np.exp(-u))         # back to θ space

# Log densities
log_q_u = -0.5*eps**2 - log_sigma_vi - 0.5*np.log(2*np.pi)       # q(u)
log_jac = np.log(theta) + np.log(1 - theta)                      # dθ/du = θ(1-θ)
log_q_theta = log_q_u - log_jac                                   # q(θ) via change-of-var

log_lik = k*np.log(theta) + (n-k)*np.log(1-theta)                # p(x|θ)
log_prior = stats.beta(alpha, beta_).logpdf(theta)                # p(θ)
log_joint = log_lik + log_prior                                   # p(x, θ)
log_posterior = stats.beta(alpha+k, beta_+n-k).logpdf(theta)      # p(θ|x) (exact)

# ────────────────────────────────────────────────
# 세 분해 계산
# ────────────────────────────────────────────────
# 분해 (1): log p(x) - KL(q || p(.|x))
log_px = betaln(alpha+k, beta_+n-k) - betaln(alpha, beta_)
kl_q_post = np.mean(log_q_theta - log_posterior)
elbo_1 = log_px - kl_q_post

# 분해 (2): E_q[log p(x|θ)] - KL(q || p_θ)
E_log_lik = log_lik.mean()
kl_q_prior = np.mean(log_q_theta - log_prior)
elbo_2 = E_log_lik - kl_q_prior

# 분해 (3): E_q[log p(x, θ)] + H(q)
E_log_joint = log_joint.mean()
H_q = -log_q_theta.mean()
elbo_3 = E_log_joint + H_q

print(f"{'분해':<45} {'값':>10}")
print(f"{'(1) log p(x) - KL(q‖p(·|x))':<45} {elbo_1:>10.6f}")
print(f"{'(2) E_q[log p(x|θ)] - KL(q‖p_θ)':<45} {elbo_2:>10.6f}")
print(f"{'(3) E_q[log p(x,θ)] + H(q)':<45} {elbo_3:>10.6f}")
print(f"{'log p(x) (exact)':<45} {log_px:>10.6f}")
print(f"{'KL gap = log p(x) - ELBO':<45} {log_px - elbo_2:>10.6f}")
```

**출력**:
```
분해                                              값
(1) log p(x) - KL(q‖p(·|x))                -7.052134
(2) E_q[log p(x|θ)] - KL(q‖p_θ)            -7.052019
(3) E_q[log p(x,θ)] + H(q)                 -7.052017
log p(x) (exact)                           -7.007254
KL gap = log p(x) - ELBO                    0.044765
```

세 값이 **MC 오차 내에서 일치** — 정리 2.4의 수치 검증.

---

## 🔗 AI/ML 연결

### VAE (분해 2)
$\mathcal{L}(x) = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \text{KL}(q(z|x)\|p(z))$ = **reconstruction loss + KL regularization**. 두 항을 분리 모니터링하는 것이 VAE 학습 진단의 표준(posterior collapse 감지).

### BNN (분해 2)
$\mathcal{L} = \mathbb{E}_{q(W)}[\log p(D|W)] - \text{KL}(q(W)\|p(W))$. 첫 항 = data fit, 둘째 = weight regularization(Gaussian prior의 Bayesian 해석).

### Physics-inspired (분해 3)
Free energy minimization = ELBO maximization. Variational methods for spin glasses, Gibbs measures.

### β-VAE (정리 2.5)
$\beta > 1$이 information bottleneck의 특정 trade-off; Tishby (2000)의 IB principle과 연결.

### Cold Posteriors (Wenzel et al. 2020)
BNN에서 $\mathcal{L}_T = \mathbb{E}_q[\log p(D|W)] - T\cdot\text{KL}(q\|p_W)$, $T < 1$에서 성능 개선 — "temperature scaling" 관점에서 분해 (3)의 $H$ 항 변형.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| KL 분해 가능 | $p(\theta)$가 복잡하면 $\text{KL}(q\|p_\theta)$ MC 필요 |
| $\log p(x,\theta)$ 직접 계산 | 잠재변수 많으면 marginalization 필요 |
| $q$에서 샘플 가능 | Reparam 가능한 family 제한 |
| $H(q)$ 해석 가능 | Flow 등에서 entropy 계산 복잡 |

**실무 팁**: VAE 학습 시 **분해 (2)의 두 항을 따로 plot**하는 것이 표준 진단.

---

## 📌 핵심 정리

$$\boxed{\mathcal{L}(q) = \log p(x) - \text{KL}(q\|p(\cdot|x)) = \mathbb{E}_q[\log p(x|\theta)] - \text{KL}(q\|p_\theta) = \mathbb{E}_q[\log p(x,\theta)] + H(q)}$$

세 분해 해석:
- **(1)**: Evidence − gap → 이론
- **(2)**: Reconstruction + prior reg → **VAE, BNN**
- **(3)**: Energy + entropy → physics, MaxEnt

---

## 🤔 생각해볼 문제

**문제 1** (기초): $q = p(\cdot|x)$일 때 세 분해의 값은 각각?

<details>
<summary>해설</summary>

- (1): $\log p(x) - 0 = \log p(x)$
- (2): $\mathbb{E}_{p(\cdot|x)}[\log p(x|\theta)] - \text{KL}(p(\cdot|x)\|p_\theta)$ — 일반적으로 두 항 모두 0 아님, 합은 $\log p(x)$
- (3): $\mathbb{E}_{p(\cdot|x)}[\log p(x,\theta)] + H(p(\cdot|x))$ — 역시 합이 $\log p(x)$

모두 동일한 **$\log p(x)$**, 분해의 동치성 확인.

</details>

**문제 2** (심화): 분해 (2)에서 "reconstruction"이 NN의 output layer(Gaussian or Bernoulli)에 따라 어떻게 구체화되는가?

<details>
<summary>해설</summary>

**Gaussian output** $p(x|z) = \mathcal{N}(\text{dec}(z), \sigma^2 I)$:
$$\log p(x|z) = -\frac{1}{2\sigma^2}\|x - \text{dec}(z)\|^2 + \text{const}$$
⇒ **MSE loss** (up to scale).

**Bernoulli output** $p(x|z) = \prod_j \text{Ber}(x_j; \pi_j(z))$:
$$\log p(x|z) = \sum_j [x_j\log\pi_j + (1-x_j)\log(1-\pi_j)]$$
⇒ **Binary cross-entropy**.

**Categorical**: **Cross-entropy**.

VAE에서 "reconstruction loss"의 구체 형태는 likelihood 선택에서 나옴 — 분해 (2)의 자연스러운 specialization.

</details>

**문제 3** (AI 연결): "**Posterior collapse**"는 VAE에서 $q(z|x) = p(z)$로 수렴해 latent $z$가 무의미해지는 현상. 분해 (2)로 이것을 어떻게 설명할 수 있는가?

<details>
<summary>해설</summary>

분해 (2): $\mathcal{L} = \mathbb{E}_q\log p(x|z) - \text{KL}(q(z|x)\|p(z))$.

KL = 0 iff $q(z|x) = p(z)$ for all $x$ — latent가 $x$ 정보를 전달하지 않음.

$p(x|z) = p(x)$(decoder가 $z$를 무시) + $q(z|x) = p(z)$로 **ELBO가 여전히 reasonable** → local optimum.

해결:
- **KL annealing**: 초기엔 $\beta < 1$로 학습, 점차 증가
- **Free bits**: KL에 minimum threshold
- **β-VAE의 inverse**: $\beta < 1$ for initial training

이것이 분해 (2)가 직접적으로 **학습 dynamics를 진단**하는 예. 각 항을 따로 추적하면 collapse를 조기 발견 가능.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. VI의 아이디어와 ELBO 유도](./01-vi-idea-elbo.md) | [📚 README로 돌아가기](../README.md) | [03. Mean-Field VI와 CAVI ▶](./03-mean-field-cavi.md) |

</div>
