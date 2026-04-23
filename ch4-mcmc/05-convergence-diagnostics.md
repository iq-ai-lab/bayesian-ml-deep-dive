# 05. 수렴 진단 — $\hat R$과 ESS

## 🎯 핵심 질문

- **Gelman-Rubin $\hat R$**의 within-variance $W$ vs between-variance $B$은 어떻게 수렴을 진단?
- $\hat R = \sqrt{\frac{N-1}{N} + \frac{B}{NW}}$의 유도와 $\hat R < 1.01$ 기준?
- **Effective Sample Size** $\text{ESS} = N/(1 + 2\sum_k \rho_k)$가 왜 autocorrelation을 보정하는가?
- MCMC 진단의 표준 체크리스트?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

MCMC가 "수렴했는지" 판단하지 못하면 결과 신뢰 불가. $\hat R$과 ESS는 **posterior inference 품질의 정량적 척도**. PyMC·Stan·ArviZ 표준 진단. BNN의 MCMC, hierarchical Bayesian analysis, clinical trial Bayesian adaptive design 모두 이 진단 체계에 의존. 잘못된 diagnostic 해석이 **gpt-prompt-engineering의 잘못된 결론**으로 이어짐.

---

## 📐 수학적 선행 조건

- [Ch4-01~04](./01-metropolis-hastings.md): MCMC 기초
- Variance decomposition (ANOVA-style)
- Autocorrelation, spectral density

---

## 📖 직관적 이해

### $\hat R$의 핵심 아이디어

여러 **chain**을 **서로 다른 초기값**에서 시작:
- **수렴 전**: chain 간 다른 위치 → between variance $B$ 큼
- **수렴 후**: 모든 chain이 posterior에서 샘플 → within = between

$\hat R = 1$에 가까우면 수렴.

### ESS의 핵심 아이디어

MCMC 샘플은 **autocorrelated** — IID보다 "정보 적음". $N$개 correlated 샘플 = $\text{ESS}$개 IID 샘플과 equivalent.

$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^\infty \rho_k}$$

$\rho_k$ = lag-$k$ autocorrelation. 강한 positive autocorrelation → ESS ↓.

### 요리 비유

- $\hat R$: "여러 부엌에서 같은 요리를 시도 — 시간이 지나면 모두 같은 결과를 내야"
- ESS: "샘플 10000개라도 similar 샘플 많으면 실제 정보는 100개 수준"

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Within-Chain Variance

$M$개 chain, 각 $N$ samples $\theta^{(m)}_t$:

$$W = \frac{1}{M}\sum_{m=1}^M s_m^2, \quad s_m^2 = \frac{1}{N-1}\sum_{t=1}^N (\theta_t^{(m)} - \bar\theta_{\cdot}^{(m)})^2$$

$\bar\theta_{\cdot}^{(m)}$ = chain $m$의 평균.

### 정의 5.2 — Between-Chain Variance

$$B = \frac{N}{M-1}\sum_{m=1}^M (\bar\theta_{\cdot}^{(m)} - \bar\theta_{\cdot\cdot})^2$$

$\bar\theta_{\cdot\cdot}$ = 전체 평균.

### 정의 5.3 — Pooled Variance Estimate

$$\hat V = \frac{N-1}{N}W + \frac{1}{N}B$$

True marginal variance의 estimator.

### 정의 5.4 — Potential Scale Reduction $\hat R$

$$\hat R = \sqrt{\frac{\hat V}{W}} = \sqrt{\frac{N-1}{N} + \frac{B}{NW}}$$

### 정의 5.5 — Autocorrelation & ESS

Lag-$k$ autocorrelation:
$$\rho_k = \frac{\text{Cov}(\theta_t, \theta_{t+k})}{\text{Var}(\theta_t)}$$

**Effective Sample Size**:
$$\text{ESS} = \frac{N}{1 + 2\sum_{k=1}^{K^*}\rho_k}$$

$K^*$ = first $k$ where $\rho_k + \rho_{k+1} < 0$ (empirical cutoff).

---

## 🔬 정리와 증명

### 정리 5.1 — $\hat R$의 한계값

**명제**: $N, M \to \infty$에서 chain이 올바른 posterior로 수렴하면:

$$\hat R \to 1$$

**증명 개요**:

수렴한 chain들의 평균이 posterior mean으로 수렴 → $B/N \to 0$ (between-chain variance가 샘플 평균의 분산).

$\hat V \to W \to \text{Var}_\pi(\theta)$.

$\hat R = \sqrt{\hat V/W} \to 1$. $\square$

**실전 기준**: $\hat R < 1.01$ (엄격), $\hat R < 1.05$ (loose). Gelman-Rubin 원문은 $1.1$ 추천했으나 현재는 더 엄격.

### 정리 5.2 — $\hat R$은 Upper Bound

**명제** (Gelman-Rubin 1992): $\hat R \geq 1$ almost always.

**증명**: $\hat V \geq W$ under reasonable chain behavior. 수렴 전엔 between-chain variance가 within-chain보다 큼. $\square$

### 정리 5.3 — Rank-Normalized $\hat R$ (Vehtari et al. 2021)

**명제**: 전통 $\hat R$이 heavy-tailed posterior에서 실패할 수 있음. 개선:

1. **Rank-transform**: $\theta^{(m)}_t \to$ rank → $\Phi^{-1}(\text{rank}/N)$ (normal)
2. **Split chains**: 각 chain을 반으로 나눠 내부 non-stationarity 감지
3. **Folded $\hat R$**: $|\theta - \text{median}|$의 $\hat R$ ← dispersion 변화 감지

**Max** of standard, folded rank-normalized $\hat R$을 report.

### 정리 5.4 — ESS의 Variance Interpretation

**명제**: Sample mean $\bar\theta_N = \frac{1}{N}\sum_t \theta_t$의 Monte Carlo 분산:

$$\text{Var}[\bar\theta_N] = \frac{\text{Var}(\theta)}{N}(1 + 2\sum_k\rho_k)$$

따라서:
$$\text{ESS} = N/(1 + 2\sum_k\rho_k)$$

**$\text{ESS}$개의 IID sample과 같은 Monte Carlo 정확도**.

**증명**:
$$\text{Var}[\bar\theta_N] = \frac{1}{N^2}\sum_{s,t}\text{Cov}(\theta_s, \theta_t) = \frac{\sigma^2}{N^2}\sum_{s,t}\rho_{|s-t|}$$

$\sum_{s,t}\rho_{|s-t|} = N + 2\sum_{k=1}^{N-1}(N-k)\rho_k \approx N(1 + 2\sum_k\rho_k)$.

$\square$

### 정리 5.5 — Tail ESS vs Bulk ESS

**명제** (Vehtari et al. 2021):

- **Bulk ESS**: 중앙부 (median, mean) inference용
- **Tail ESS**: tail quantiles (5%, 95%) — rank-transformed ESS

Quantile estimation에선 bulk보다 tail이 중요. 실전 기준:
$$\text{bulk ESS} > 400, \quad \text{tail ESS} > 400$$

---

## 💻 구현

```python
import numpy as np
import matplotlib.pyplot as plt
import arviz as az

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# 간단한 MH on Beta posterior (4 chains)
# ────────────────────────────────────────────────
def log_pi(theta):
    if theta <= 0 or theta >= 1: return -np.inf
    return 8*np.log(theta) + 4*np.log(1-theta)  # Beta(9, 5)

def mh_chain(init, n_iter, step, seed):
    rng_l = np.random.default_rng(seed)
    samples = np.zeros(n_iter)
    theta = init
    for t in range(n_iter):
        theta_prop = theta + step*rng_l.standard_normal()
        if np.log(rng_l.random()) < log_pi(theta_prop) - log_pi(theta):
            theta = theta_prop
        samples[t] = theta
    return samples

n_iter = 5000
inits = [0.1, 0.3, 0.7, 0.9]
chains = np.array([mh_chain(init, n_iter, 0.15, seed=i) 
                   for i, init in enumerate(inits)])
# shape: (4 chains, 5000 samples)

burn = 1000
chains_post = chains[:, burn:]  # (M=4, N=4000)

# ────────────────────────────────────────────────
# R-hat 계산 (직접)
# ────────────────────────────────────────────────
M, N = chains_post.shape
chain_means = chains_post.mean(axis=1)         # (M,)
overall_mean = chains_post.mean()
chain_vars = chains_post.var(axis=1, ddof=1)   # (M,)

W = chain_vars.mean()
B = N * ((chain_means - overall_mean)**2).sum() / (M - 1)
V_hat = (N - 1)/N * W + B/N
R_hat = np.sqrt(V_hat / W)
print(f"W = {W:.5f}")
print(f"B = {B:.5f}")
print(f"R-hat = {R_hat:.4f}")
print(f"R-hat < 1.01? {R_hat < 1.01}")

# ────────────────────────────────────────────────
# ESS 계산
# ────────────────────────────────────────────────
from statsmodels.tsa.stattools import acf
rhos = acf(chains_post[0], nlags=200)
# First negative lag cutoff
cutoff = np.where(rhos[1:] + rhos[2:] < 0)[0]
K = cutoff[0] if len(cutoff) > 0 else 200
ess = N / (1 + 2*rhos[1:K+1].sum())
print(f"\nESS (chain 0) = {ess:.0f}")
print(f"Autocorrelation time = {1 + 2*rhos[1:K+1].sum():.2f}")

# ────────────────────────────────────────────────
# ArviZ는 이 모든 것을 자동으로
# ────────────────────────────────────────────────
idata = az.from_dict({'theta': chains_post})
print("\nArviZ summary:")
print(az.summary(idata, var_names=['theta']))
# columns: mean, sd, hdi_3%, hdi_97%, mcse_mean, mcse_sd, ess_bulk, ess_tail, r_hat

# ────────────────────────────────────────────────
# Visualization
# ────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Trace plot - all chains
for m in range(M):
    axes[0, 0].plot(chains[m, :1500], lw=0.5, alpha=0.7, label=f'chain {m}')
axes[0, 0].axvline(burn, color='r', ls='--', label='burn-in end')
axes[0, 0].set_title('All chains (different inits → same posterior)')
axes[0, 0].legend(); axes[0, 0].grid(alpha=0.3)

# Running R-hat
r_hats_traj = []
window = 500
for t in range(window, N, 100):
    c_slice = chains_post[:, :t]
    cm = c_slice.mean(axis=1)
    om = c_slice.mean()
    W_t = c_slice.var(axis=1, ddof=1).mean()
    B_t = t * ((cm - om)**2).sum() / (M - 1)
    V_t = (t - 1)/t * W_t + B_t/t
    r_hats_traj.append(np.sqrt(V_t/W_t))
axes[0, 1].plot(np.arange(window, N, 100), r_hats_traj, 'o-')
axes[0, 1].axhline(1.01, color='r', ls='--', label=r'$\hat R = 1.01$')
axes[0, 1].set_title(r'$\hat R$ vs sample size (should → 1)')
axes[0, 1].legend(); axes[0, 1].grid(alpha=0.3)

# Autocorrelation
axes[1, 0].stem(rhos[:60], basefmt=' ')
axes[1, 0].axhline(0, color='k', lw=0.5)
axes[1, 0].set_title('Autocorrelation (chain 0)')
axes[1, 0].set_xlabel('lag'); axes[1, 0].grid(alpha=0.3)

# Histogram (final)
axes[1, 1].hist(chains_post.flatten(), bins=60, density=True, alpha=0.5)
theta_grid = np.linspace(0.2, 0.95, 200)
from scipy import stats
axes[1, 1].plot(theta_grid, stats.beta(9, 5).pdf(theta_grid), 'r-', lw=2, label='True Beta(9,5)')
axes[1, 1].legend(); axes[1, 1].set_title('All chains combined')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig('mcmc_diagnostics.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Stan / PyMC Workflow
모든 sampling run 후 **자동 진단** — $\hat R > 1.01$ warning.

### Hierarchical Model
Funnel posterior에서 $\hat R$ 감지 → 재매개변수화 motivation.

### BNN MCMC
SGLD·SGHMC의 수렴 진단도 $\hat R$, ESS 기반.

### A/B Testing Bayesian
실시간 의사결정에 posterior sample 사용 → ESS가 작으면 **결정 불확실성 과소추정**.

### Model Comparison
Bayes factor 계산의 안정성 → ESS로 검증.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Multiple chains 있음 | Single chain에선 $\hat R$ 의미 없음 |
| Chain이 진짜 stationary | 수렴 전에도 $\hat R \approx 1$ 가능(multi-mode 한쪽 trap) |
| Finite variance | Heavy-tail에선 rank-normalized 버전 필수 |
| Gaussian-like | 극단 skew에선 quantile-based ESS |

**실무 표준**:
1. $\geq 4$ chains, 서로 다른 inits
2. Warmup + sampling 각 1000+
3. $\hat R < 1.01$ for every parameter
4. Bulk + tail ESS $> 400$
5. No divergent transitions (HMC)
6. Trace plot 육안 점검
7. **Rank plot**: chain별 rank 분포 uniform해야

---

## 📌 핵심 정리

$$\boxed{\hat R = \sqrt{\frac{N-1}{N} + \frac{B}{NW}}, \quad \hat R \to 1}$$

$$\boxed{\text{ESS} = \frac{N}{1 + 2\sum_k\rho_k}}$$

체크리스트:
- $\hat R < 1.01$ (bulk + folded + rank-normalized)
- Bulk + tail ESS $> 400$
- Trace plot, rank plot OK
- HMC: no divergences

---

## 🤔 생각해볼 문제

**문제 1** (기초): $\hat R = 1.5$일 때 어떤 의미?

<details>
<summary>해설</summary>

Chains가 **서로 다른 지역에 있음** → still converging or multimodal trap. Pooled variance가 within보다 **2.25× 큼**.

조치:
- Warmup 증가
- 초기값 더 분산
- 모델 재매개변수화
- Multimodal 가능성 조사

절대 **이 sample로 inference 하지 말 것**.

</details>

**문제 2** (심화): 10,000 샘플 중 ESS = 200이면 무엇을 의미?

<details>
<summary>해설</summary>

유효 정보 = **200 IID samples 수준**. Autocorrelation time $\approx 50$ (샘플 50개 = IID 1개).

- 중앙 추정 (평균): OK
- Quantile/tail: 부족 → 더 많이 run
- 95% CI 정밀도: $1/\sqrt{200}$ = ±7% relative — credible interval이 wide

개선:
- Thinning (보통 불필요 — 그냥 더 run)
- Step size/trajectory 튜닝
- 더 긴 sampling

**400** ESS가 최소 기준 (Vehtari et al. 2021).

</details>

**문제 3** (AI 연결): NN loss landscape의 mode가 여러 개일 때, $\hat R$ 진단의 한계는?

<details>
<summary>해설</summary>

**문제**: 여러 modes가 거의 같은 마진 확률일 때:
- 모든 chain이 같은 모드로 빠지면 $\hat R = 1$ ← **false positive** (수렴한 것처럼 보임)
- 다른 모드로 빠지면 $\hat R \gg 1$ ← 정확히 진단

실전: BNN에서 단일 mode에 갇히는 것이 흔 — $\hat R = 1$이어도 **posterior의 한 부분만**.

**해결**:
- **Multi-start** inits (deep ensembles와 유사)
- **Tempering/replica exchange** (parallel tempering MCMC)
- **SG-MCMC** variants의 exploration

Ch5-05 SGLD·SWAG 같은 **BNN-specific methods**가 이 문제를 완전 해결하진 못하지만 **approximation**으로 우회.

Bayesian DL의 open problem — "**full posterior exploration는 여전히 어려움**".

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. No-U-Turn Sampler](./04-nuts.md) | [📚 README로 돌아가기](../README.md) | [06. MCMC의 한계와 VI와의 비교 ▶](./06-mcmc-vs-vi.md) |

</div>
