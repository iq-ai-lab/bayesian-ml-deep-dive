# 06. MCMC의 한계와 VI와의 비교

## 🎯 핵심 질문

- MCMC는 어떤 경우에 **실패**하는가? Multimodal·high-dim·tight-correlation?
- **VI vs MCMC**의 정확도 vs 속도 trade-off를 어떻게 정량화하는가?
- 문제 특성에 따라 어느 방법을 선택해야 하는가?
- 둘을 **결합**하는 hybrid 방법은?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

"어느 방법을 써야 하나"는 Bayesian workflow의 첫 결정. 잘못 선택하면 며칠 낭비 또는 잘못된 결론. 산업계 대부분의 Bayesian DL은 VI (scalability), 중소 science·research는 MCMC (accuracy). 둘을 **정확히 이해하고 선택**하는 것이 **실전 Bayesian 능력의 차이**.

---

## 📐 수학적 선행 조건

- [Ch2](../ch2-variational-inference/01-vi-idea-elbo.md): VI 전체
- [Ch4-01~05](./01-metropolis-hastings.md): MCMC 전체

---

## 📖 직관적 이해

### 비교표

| 측면 | VI | MCMC |
|------|----|----|
| 출력 | $q_\phi(\theta)$ (분포 객체) | samples $\{\theta^{(t)}\}$ |
| 속도 | **빠름** (gradient-based opt) | 느림 (sequential sampling) |
| 정확도 | Family 제약 → approximation gap | **Asymptotically exact** |
| 고차원 | 상대적으로 견고 | 느려짐 (HMC는 완화) |
| Posterior correlation | Mean-field는 놓침 | 정확 |
| Multimodal | Mode-seeking (reverse KL) | Mode-hopping 어려움 |
| 병렬 | NN 학습 자연 | Multi-chain만 |
| Streaming | SVI 가능 | 어려움 |
| 불확실성 정량화 | Variance underestimate 위험 | 정확 |
| 진단 | ELBO trace | $\hat R$, ESS |

### 언제 VI

- **대규모 데이터** (모델 내 forward pass로 전체 loss 계산 가능)
- **NN latent model** (VAE, BNN)
- **실시간 inference** (amortized encoder)
- **Point estimate + 대략적 variance로 충분**
- **연구/prototype 빠른 반복**

### 언제 MCMC

- **정확한 posterior**가 중요 (clinical decision, scientific analysis)
- **Model comparison** (Bayes factor → evidence 정확도 필요)
- **Moderate dimensionality** ($d < 10^4$)
- **Hierarchical model** (correlation 많음)
- **Interpretable uncertainty** 필요

### 요리 비유

- VI: "메뉴 몇 개만 골라 리허설 — 빠르지만 메뉴 제약"
- MCMC: "모든 메뉴를 충분히 시도 — 느리지만 포괄적"

---

## ✏️ 엄밀한 정의 / 비교 축

### 정의 6.1 — Approximation Error Metrics

- **$d_{TV}(\hat q, p)$**: total variation
- **KL divergence**: 방향별
- **Posterior quantile coverage**: 실전적

### 정의 6.2 — Computational Cost

- **VI**: $O(T \cdot N_{batch} \cdot \text{NN forward})$, $T$ = epochs
- **MCMC**: $O(T \cdot \text{likelihood eval} \cdot \text{ESS efficiency})$

### 정의 6.3 — Hybrid Methods

- **VI-init → MCMC**: VI로 초기 posterior → MCMC로 refine
- **Stochastic gradient Langevin dynamics (SGLD)**: SGD + Langevin noise (Ch5-05)
- **Riemannian HMC with VI pre-conditioning**: VI로 $M$ 추정
- **Parallel MCMC + variational**: IWAE (Ch3-05)가 그 예

---

## 🔬 정리와 경험 법칙

### 정리 6.1 — 점근적 Exactness vs 유한 Error

**명제**:

**MCMC**: $T \to \infty$에서 sample이 $\pi$로 수렴. 유한 $T$에선 $O(1/\sqrt{T})$ 오차 + bias.

**VI**: 유한 시간에 수렴하지만 **approximation gap $\geq 0$** 영구적 남음 (family 제약).

**실전 귀결**: VI는 **"빠르고 biased"**, MCMC는 **"느리고 unbiased"**.

### 정리 6.2 — Multimodal Posterior의 실패 모드

**VI (reverse KL)**: Mode-seeking — **한 mode에 집중**, 다른 mode 무시. "Zero forcing": $q$ small where $p$ is zero.

**MCMC (single chain)**: Mode 간 이동 **지수적으로 느림** (energy barrier 통과 필요). 단일 chain이 multiple mode 탐색 실패.

**실전 해결**:
- VI: Mixture of Gaussians as $q$, Normalizing flow
- MCMC: Parallel tempering, replica exchange, deep ensembles
- Hybrid: VI로 **mode 위치 찾기** + chain별로 MCMC

### 정리 6.3 — Scalability Comparison

| Component | VI | MCMC |
|-----------|----|----|
| $N$ (data) | Mini-batch → $O(\sqrt N)$ / epoch | Full data / step |
| $d$ (params) | NN scalable | HMC: $O(d^{5/4})$; RW-MH: $O(d)$ mixing time |
| $|\mathcal{D}|$ types | Mixed (image, text) via NN | Often needs custom model |

### 정리 6.4 — 진단 비교

**VI**:
- ELBO trace (수렴하면 plateau)
- Posterior predictive 시각화
- **KL gap 추정 어려움** ($\log p(x)$ 모름)

**MCMC**:
- $\hat R$, ESS (Ch4-05)
- Trace plot
- Divergences (HMC)
- **오히려 더 rigorous** 진단 가능

### 정리 6.5 — 사용 선택 가이드 (heuristic)

| 상황 | 추천 |
|------|------|
| $N = 10^6$, NN latent | VAE/BNN VI |
| $N = 10^3$, $d = 20$ hierarchical | NUTS |
| Bayesian optimization | GP + exact (Ch6) |
| Real-time inference | Amortized VI |
| Clinical decision | MCMC (reliability) |
| Topic model at scale | SVI (LDA online) |
| Rare-event posterior | MCMC with tempering |

---

## 💻 동일 문제에 두 방법 비교

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pymc as pm

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Problem: 2-component GMM posterior (bimodal param)
# data ~ 0.5 N(θ, 1) + 0.5 N(-θ, 1), θ ~ N(0, 10)
# Posterior is bimodal because of ±θ symmetry
# ────────────────────────────────────────────────
true_theta = 2.0
N = 50
data = rng.choice([-1, 1], N) * true_theta + rng.standard_normal(N)

def log_lik_gmm(theta, data):
    lik = 0.5*stats.norm(theta, 1).pdf(data) + 0.5*stats.norm(-theta, 1).pdf(data)
    return np.log(lik).sum()

def log_post(theta, data):
    return log_lik_gmm(theta, data) + stats.norm(0, 10).logpdf(theta)

theta_grid = np.linspace(-5, 5, 400)
post_unnorm = np.exp([log_post(t, data) for t in theta_grid])
post_true = post_unnorm / np.trapz(post_unnorm, theta_grid)

# ────────────────────────────────────────────────
# Method 1: NUTS (via PyMC)
# ────────────────────────────────────────────────
with pm.Model() as model:
    theta = pm.Normal('theta', 0, 10)
    data_dist = pm.Mixture('data', w=np.ones(2)/2,
                           comp_dists=[pm.Normal.dist(theta, 1),
                                       pm.Normal.dist(-theta, 1)],
                           observed=data)
    idata = pm.sample(2000, tune=1000, chains=4, random_seed=0, progressbar=False)

mcmc_samples = idata.posterior['theta'].values.flatten()

# ────────────────────────────────────────────────
# Method 2: VI (mean-field Gaussian)
# ────────────────────────────────────────────────
from scipy.optimize import minimize
def neg_elbo(params, data, n_mc=500, seed=0):
    rng_l = np.random.default_rng(seed)
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    eps = rng_l.standard_normal(n_mc)
    theta_samples = mu + sigma*eps
    log_p = np.array([log_post(t, data) for t in theta_samples])
    log_q = -0.5*eps**2 - log_sigma - 0.5*np.log(2*np.pi)
    return -(log_p - log_q).mean()

result = minimize(neg_elbo, [0.5, 0.0], args=(data,), method='Nelder-Mead')
mu_vi, log_sigma_vi = result.x
sigma_vi = np.exp(log_sigma_vi)

# ────────────────────────────────────────────────
# Compare
# ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(theta_grid, post_true, 'k-', lw=2.5, label='True posterior')
ax.hist(mcmc_samples, bins=50, density=True, alpha=0.5, label=f'NUTS MCMC')
ax.plot(theta_grid, stats.norm(mu_vi, sigma_vi).pdf(theta_grid), 'r-', lw=2,
        label=f'VI: N({mu_vi:.2f}, {sigma_vi:.2f}²)')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel('density')
ax.set_title(f'Bimodal posterior: NUTS captures both modes, VI collapses to one')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('vi_vs_mcmc.png', dpi=150); plt.show()

print(f"VI:   μ={mu_vi:.3f}, σ={sigma_vi:.3f}")
print(f"MCMC: mean={mcmc_samples.mean():.3f}, std={mcmc_samples.std():.3f}")
print(f"True: bimodal ±{true_theta}")
```

**예상 결과**: MCMC는 ±2 두 mode 모두 capture, VI는 한 mode에 집중 (또는 사이 공간).

---

## 🔗 AI/ML 연결

### VAE = Amortized VI, DDPM VLB = Hierarchical VI
대규모 생성모델은 거의 모두 VI (MCMC는 확장 불가).

### GP with HMC
Kernel hyperparameter posterior에 NUTS (Ch6).

### Hybrid MCMC-VI
SNLE(Papamakarios 2019), Neural posterior estimation — VI로 초기, MCMC refine.

### Normalizing Flow as $q$
VI의 family 확장 → MCMC 수준 표현력 (Ch3-03).

### Deep Ensembles
여러 NN을 독립 학습 → "discrete MCMC" — multi-mode 탐색에 효과.

---

## ⚖️ 선택 가이드 (Implementation Checklist)

| 질문 | VI 선호 if | MCMC 선호 if |
|------|------------|--------------|
| 데이터 크기 $N$ | $N > 10^5$ | $N < 10^4$ |
| 모델 dim $d$ | $d > 10^3$ | $d < 10^3$ |
| Latent 사용 | NN-based | Discrete or structured |
| 정확도 우선 | No | Yes |
| 실시간 deploy | Yes | No |
| Bayes factor | No (VI evidence 어려움) | Yes |
| Exploratory | Yes (빠른 iteration) | No |
| 기존 모델 wrap | Stan/PyMC/NumPyro | MCMC 자동 |

---

## 📌 핵심 정리

$$\boxed{\text{VI: 빠르고 biased} \quad \text{vs} \quad \text{MCMC: 느리고 unbiased}}$$

핵심:
- VI는 **approximation gap** 영구
- MCMC는 **mixing time** 문제
- Multimodal: 둘 다 어려움, MCMC가 원리상 더 나음
- 산업 대규모: VI가 사실상 유일
- 과학 연구: MCMC가 표준

---

## 🤔 생각해볼 문제

**문제 1** (기초): ELBO 값만으로 posterior 품질을 판단할 수 있나?

<details>
<summary>해설</summary>

**ELBO = $\log p(x) - \text{KL}$**. 두 가지가 섞임:
- $\log p(x)$: 데이터에 model fit
- $-\text{KL}$: approximation quality

**같은 ELBO**여도 model 다르면 해석 다름. 절대값보단 **같은 model의 다른 $q$** 비교 시 유용.

**정확한 품질 평가**: held-out data의 **log-likelihood** 추정 (IWAE, AIS) 또는 **posterior predictive** check.

</details>

**문제 2** (심화): "VI의 variance underestimate"가 **downstream decision**에 어떻게 영향?

<details>
<summary>해설</summary>

**예**: 의료 진단 probability 0.85, VI variance 0.01, 실제 variance 0.05.

- VI: "확신 있음, treat"
- 실제: "uncertainty 크므로 추가 검사"

**VI가 false confidence** → 과잉 신뢰 의사결정.

Risk-aware 응용 (금융, 의료, 자율주행):
- VI 결과의 variance를 **empirical calibration**으로 조정
- Critical decision엔 MCMC 또는 ensemble 검증
- Reverse KL의 "mode-seek" 특성 인식 → posterior tail 중요한 경우 VI 위험

정리 3.4 (Ch2-03)의 theoretical bias를 실전에서 조심.

</details>

**문제 3** (AI 연결): GPT·LLM의 inference는 본질적으로 point estimate (MLE/MAP). 이것이 **VI·MCMC와 어떻게 관련**되나?

<details>
<summary>해설</summary>

LLM은 **frequentist point estimator** — weights는 단일 값. Bayesian이 아님.

그러나:
- **Dropout at inference**: MC Dropout → approximate VI (Ch5-04)
- **Ensemble of fine-tuned models**: deep ensemble = discrete MCMC-like
- **LoRA / PEFT with Bayesian prior**: Bayesian fine-tuning 연구

**Scaling limit**: 모델이 커질수록 trained model ≈ posterior mode (Laplace 관점). 실제 uncertainty quantification은 여전히 open problem.

**최근 연구**:
- **SWAG for LLM**: 훈련 trajectory로 Bayesian (Ch5-05)
- **Bayesian LoRA**: adapter 파라미터에 Bayesian 추론
- **Test-time consistency**: 같은 prompt의 multiple generation → sampling-based uncertainty

LLM을 Bayesian으로 만드는 것이 **현재 연구 frontier**. VI/MCMC 이해가 이 분야의 기초.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 05. 수렴 진단 — R̂과 ESS](./05-convergence-diagnostics.md) | [📚 README로 돌아가기](../README.md) | [Ch5-01. BNN의 수학적 정식화 ▶](../ch5-bayesian-nn/01-bnn-formulation.md) |

</div>
