# 05. SWAG와 SGD의 Bayesian 해석

## 🎯 핵심 질문

- **SWAG** (Maddox et al. 2019): SGD 궤적의 mean과 covariance를 어떻게 **Gaussian posterior** 근사로 만드는가?
- SGD가 **implicit Bayesian**이라는 Mandt et al. (2017)의 주장은?
- **SGLD** $W_{k+1} = W_k + \frac{\eta}{2}\nabla\log p + \sqrt{\eta}\xi$가 Langevin SDE의 이산화이고 정상분포 = posterior인 이유?
- 이 방법들이 MC Dropout이나 BBB 대비 장단점?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**SGD 트레이닝만 했는데 Bayesian이 된다**는 것이 SWAG의 매력. Fine-tuning 지루한 BBB나 dropout 필요 없이 **이미 학습된 궤적**을 재사용. SGLD는 Welling & Teh 2011의 고전 — SGD + noise = Bayesian MCMC. 대규모 BNN의 실전 posterior inference.

---

## 📐 수학적 선행 조건

- [Ch5-01~04](./01-bnn-formulation.md): BNN 전체
- [SDE Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive) Ch4 Fokker-Planck: Langevin의 정상분포
- SGD optimization dynamics
- Multivariate Gaussian

---

## 📖 직관적 이해

### SWAG의 아이디어

SGD로 학습 중 **일정 주기로 weight snapshot** 저장:
$$W_1, W_2, \ldots, W_K$$

이 궤적의:
- **Mean**: $\bar W = \frac{1}{K}\sum W_k$ (SWA — Stochastic Weight Averaging)
- **Covariance**: $\hat\Sigma$ (low-rank + diagonal 근사)

⇒ Posterior $\mathcal{N}(\bar W, \hat\Sigma)$.

### SGD = Implicit Bayesian

Mandt et al. (2017): Constant learning rate SGD의 stationary distribution이 **local posterior의 approximation**. Langevin noise이 mini-batch gradient noise에 "내장".

### SGLD

$$W_{k+1} = W_k + \frac{\eta}{2}\nabla\log p(W_k|D) + \sqrt{\eta}\xi_k, \quad \xi_k\sim\mathcal{N}(0, I)$$

Langevin SDE $dW = \frac{1}{2}\nabla\log\pi\,dt + dB_t$의 Euler-Maruyama 이산화. 정상분포 = $\pi = p(W|D)$ (SDE Ch4-03).

Data subsampling 가능: $\nabla\log p$ 대신 mini-batch approximation.

### 요리 비유

- SWAG: "훈련 중 **여러 시점의 요리 기록** → 요리사 분포 추정"
- SGLD: "SGD + 약간의 random jitter → 궤적이 posterior 샘플"

---

## ✏️ 엄밀한 정의

### 정의 5.1 — Stochastic Weight Averaging (SWA)

$T$ SGD iterations의 후반부 $k \in [T_0, T]$에서:
$$\bar W_{SWA} = \frac{1}{T - T_0}\sum_{k=T_0}^T W_k$$

### 정의 5.2 — SWAG

SWAG extends SWA with:
- **Diagonal variance**: $\hat\sigma_{ii}^2 = \text{sample variance of } W_{k,i}$
- **Low-rank factor**: $\hat D = $ deviation matrix (K columns)

Posterior approximation:
$$p(W|D) \approx \mathcal{N}\left(\bar W, \frac{1}{2}(\hat\Sigma_{\text{diag}} + \hat D \hat D^T/(K-1))\right)$$

### 정의 5.3 — Langevin Dynamics (Continuous)

$$dW_t = \frac{1}{2}\nabla\log p(W_t|D)dt + dB_t$$

Stationary $\pi(W) = p(W|D)$ (Ch4 SDE 레포 정리 4.3).

### 정의 5.4 — SGLD

Euler-Maruyama 이산화:
$$W_{k+1} = W_k + \frac{\eta_k}{2}\nabla\log p(W_k|D) + \sqrt{\eta_k}\xi_k$$

$\eta_k \to 0$ (decreasing step) + Robbins-Monro ($\sum\eta_k = \infty, \sum\eta_k^2 < \infty$) → correct convergence.

### 정의 5.5 — Mini-Batch SGLD

Full $\nabla\log p(W|D) = \sum_i\nabla\log p(y_i|x_i, W) + \nabla\log p(W)$ 대신 batch:
$$\tilde\nabla = \frac{N}{|B|}\sum_{i\in B}\nabla\log p(y_i|x_i, W) + \nabla\log p(W)$$

Stochastic estimator.

---

## 🔬 정리와 증명

### 정리 5.1 — SWA의 Flat Minimum 수렴

**명제** (Izmailov et al. 2018): SGD가 SWA point $\bar W$로 수렴하면 test error가 **개별 SGD point보다 낮음**. 이는 $\bar W$가 "**flat minimum**"에 있음을 시사.

**증명 스케치** (empirical + theoretical intuition): Constant LR SGD는 loss landscape의 flat region으로 수렴 경향 (Keskar 2017) → 궤적 평균이 center of flat region → generalization 개선. $\square$

### 정리 5.2 — SGD = Langevin (Constant LR)

**명제** (Mandt et al. 2017): Constant LR SGD의 stationary distribution이 특정 조건 하에서:

$$\pi_{\text{SGD}}(W) \propto \exp(-L(W)/T_{\text{eff}})$$

with effective temperature $T_{\text{eff}} \propto \eta \cdot (\text{gradient noise variance})/(\text{batch size})$.

**증명 스케치**: Mini-batch gradient noise $\epsilon$가 Gaussian approximation 하 $\eta\epsilon$이 Langevin noise에 대응. $T_{\text{eff}} < 1$이면 $\pi_{SGD}$가 posterior 근처, $T_{\text{eff}} = 1$이면 exactly posterior. $\square$

**Practical implication**: **SGD 궤적 = Bayesian posterior의 MCMC 샘플** (approximately).

### 정리 5.3 — SGLD 정상분포 = Posterior

**명제**: Langevin SDE $dW = \frac{1}{2}\nabla\log p(W|D)dt + dB_t$의 stationary:

$$\pi(W) = p(W|D)$$

**증명**: SDE 레포 Ch4 정리 4.3 직접 적용. Fokker-Planck:
$$\partial_t p = -\nabla\cdot(\frac{1}{2}\nabla\log\pi \cdot p) + \frac{1}{2}\Delta p$$

$p = \pi$ 대입 → 양변 0 (stationary). $\square$

SGLD는 이 SDE의 Euler-Maruyama — $\eta \to 0$에서 SDE로 수렴.

### 정리 5.4 — SGLD의 Bias-Variance

**명제** (Welling & Teh 2011):

- $\eta$ 상수: bias $O(\eta)$
- $\eta_k = k^{-1/3}$: unbiased asymptotic, variance slow
- Mini-batch noise: bias $O(\eta\sigma_{\text{batch}}^2)$

**실전**: Constant $\eta = 10^{-5}$ ~ $10^{-4}$ with burnin sufficient.

### 정리 5.5 — SWAG Low-Rank Covariance

**명제**: Weight snapshot $W_k - \bar W$의 **top-$K$ principal directions**로 covariance 근사:

$$\hat\Sigma_{\text{low-rank}} = \frac{1}{K-1}\sum_k (W_k - \bar W)(W_k - \bar W)^T$$

원래 full $d \times d$ (저장 불가) 대신 $d \times K$ deviation matrix만 저장 ($K = 20$ typical).

Combined with diagonal:
$$\hat\Sigma = \frac{1}{2}(\hat\Sigma_{\text{diag}} + \hat\Sigma_{\text{low-rank}})$$

---

## 💻 PyTorch 구현 — SWAG

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SWAG:
    def __init__(self, model, max_num_models=20):
        self.model = model
        self.mean = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.sq_mean = {k: torch.zeros_like(v) for k, v in model.state_dict().items()}
        self.deviations = []  # list of (state_dict delta) snapshots
        self.max = max_num_models
        self.n = 0

    def collect(self):
        """Call during/after SGD training at checkpoints."""
        self.n += 1
        alpha = 1.0 / self.n
        for k, v in self.model.state_dict().items():
            self.mean[k] = (1 - alpha)*self.mean[k] + alpha*v
            self.sq_mean[k] = (1 - alpha)*self.sq_mean[k] + alpha*v**2
        # Deviation
        dev = {k: v - self.mean[k] for k, v in self.model.state_dict().items()}
        self.deviations.append(dev)
        if len(self.deviations) > self.max:
            self.deviations.pop(0)

    def sample(self, scale=1.0):
        """Sample weights from SWAG Gaussian posterior."""
        sampled = {}
        z1 = torch.randn(1)  # shared for diagonal
        z2 = torch.randn(len(self.deviations))  # for low-rank
        for k in self.mean:
            diag_var = (self.sq_mean[k] - self.mean[k]**2).clamp(min=1e-10)
            d = torch.sqrt(0.5*diag_var) * torch.randn_like(self.mean[k])
            lowrank = torch.zeros_like(self.mean[k])
            for i, dev in enumerate(self.deviations):
                lowrank += z2[i] * dev[k] / np.sqrt(2*(len(self.deviations) - 1))
            sampled[k] = self.mean[k] + scale*(d + lowrank)
        return sampled

    def predict(self, x, n_samples=50):
        """Monte Carlo predictive."""
        preds = []
        for _ in range(n_samples):
            sampled = self.sample()
            self.model.load_state_dict(sampled)
            with torch.no_grad():
                preds.append(self.model(x).numpy())
        return np.array(preds)

# 사용 예:
# 1. 표준 SGD 학습 (초기 T0 epochs)
# 2. 이후 매 epoch마다 swag.collect()
# 3. 평가 시 swag.predict(x_test)

# ────────────────────────────────────────────────
# SGLD 구현 (much simpler)
# ────────────────────────────────────────────────
def sgld_step(model, x_batch, y_batch, eta, N_total):
    # Scaled gradient: N * batch loss
    pred = model(x_batch)
    loss = 0.5*((y_batch - pred)**2).sum()  # Gaussian likelihood
    reg = 0.5*sum((p**2).sum() for p in model.parameters())  # Gaussian prior
    log_post = -(N_total / len(x_batch)) * loss - reg
    # Gradient of log posterior
    log_post.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data + 0.5*eta*p.grad + np.sqrt(eta)*torch.randn_like(p)
            p.grad = None
    return model

# 사용:
# SGD 학습 후, sgld step을 몇 천 번 실행하여 posterior samples 수집.
```

---

## 🔗 AI/ML 연결

### Generalization
SWA 자체가 test accuracy 개선 (Izmailov 2018) — flat minimum hypothesis.

### Bayesian LLM
SWAG over LLM fine-tuning trajectory → Bayesian LoRA, calibrated reward model.

### SGD Bayesian Interpretation
Constant-lr SGD ~ Langevin의 stationary → **"무의식적 Bayesian 학습"**.

### SG-MCMC
SGHMC (Chen et al. 2014), SG-NHT (Ding et al. 2014), NUTS-style SG-MCMC 등 발전.

### Connection to SDE
Ch4 SDE 레포의 Langevin dynamics가 여기서 직접 응용 — Forward SDE → Bayesian posterior sampling.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| SGD 궤적이 posterior 탐색 | Local (single-mode) only |
| Gaussian posterior 근사 | Skewed/multi-modal은 놓침 |
| SGLD: step size small enough | 너무 크면 bias, 너무 작으면 mixing 느림 |
| SGLD mini-batch noise 제어 | Batch size 너무 작으면 noise 과다 |
| Constant-lr SGD 해석 | 실전 lr schedule 있으면 이론 수정 |

**실무 팁**:
- **SWAG**: 이미 학습된 모델에 post-hoc 적용 가능 (extra few epochs 쌓기)
- **SGLD**: burn-in 충분히 (전체 iter의 50%)
- **Combine**: SWAG mean을 init으로 SGLD → 빠른 posterior exploration

---

## 📌 핵심 정리

$$\boxed{\text{SWAG: } q(W) = \mathcal{N}(\bar W_{SGD}, \hat\Sigma_{\text{low-rank + diag}})}$$

$$\boxed{\text{SGLD: } W_{k+1} = W_k + \frac{\eta}{2}\nabla\log p(W|D) + \sqrt{\eta}\xi_k}$$

핵심:
- SWAG = SGD trajectory의 Gaussian fit
- SGLD = Langevin discretization, stationary = posterior
- Free (re-use SGD) + local uncertainty
- Multimodal·정확성엔 한계 — deep ensembles 병용

---

## 🤔 생각해볼 문제

**문제 1** (기초): Langevin SDE에서 $dW = \frac{1}{2}\nabla\log p + dB$의 "1/2"은 어디서 오는가?

<details>
<summary>해설</summary>

Fokker-Planck의 **diffusion coefficient**와 맞추기 위함:

$$\partial_t p = -\nabla\cdot(bp) + \frac{1}{2}\nabla\cdot(\sigma\sigma^T\nabla p)$$

$\sigma\sigma^T = I$ (standard BM), drift $b = \frac{1}{2}\nabla\log\pi$.

Stationary $p = \pi$ 확인:
$$-\nabla\cdot(\frac{1}{2}\nabla\log\pi \cdot \pi) + \frac{1}{2}\nabla^2\pi = -\frac{1}{2}\nabla\cdot(\nabla\pi) + \frac{1}{2}\nabla^2\pi = 0 \checkmark$$

**1/2**이 없으면 stationary가 $\pi^2$ 등으로 바뀜. 물리학의 Langevin은 보통 $dW = -\nabla U + \sqrt 2\,dB$ form (same stationary via algebra).

</details>

**문제 2** (심화): SWAG의 "low-rank + diagonal"이 왜 full covariance보다 **더 좋을 수 있나**?

<details>
<summary>해설</summary>

**Full $d \times d$ covariance**:
- 저장 $O(d^2)$ — $d = 10^6$에선 불가
- $K \ll d$ 궤적 샘플로는 **rank-$K$만 복원 가능** → full이 대부분 **zero (underdetermined)**

**Low-rank + diag**:
- **Top-$K$ directions**: 궤적이 실제로 탐색한 방향 (유의미한 curvature)
- **Diagonal**: residual uncertainty
- 저장 $O(dK + d)$

즉 full이 **information-poor** (단 $K$ samples로 derived), low-rank는 **structured**. Overfit 덜함 + computationally feasible.

실전 $K = 20$ 정도면 BNN 성능 좋음.

</details>

**문제 3** (AI 연결): SGD-trained LLM에 SWAG을 적용하여 "Bayesian GPT"를 만든다면 어떤 실전 이슈?

<details>
<summary>해설</summary>

**이론**: Fine-tuning trajectory의 마지막 $K$ epochs 저장 → SWAG Gaussian.

**실전 이슈**:
1. **Memory**: $10^{11}$ params × $K = 20$ snapshots = $2 \times 10^{12}$ floats → 페타바이트. **Impossible**.
2. **Solution**:
   - **LoRA SWAG**: adapter만 ($10^6$ × 20 = OK)
   - **Per-layer SWAG**: layer별 저장 schedule
   - **Rank compression**: snapshot을 low-rank projection 저장
3. **Checkpoint selection**: 학습이 plateau에 도달한 후 수집 → meaningful exploration
4. **Inference cost**: $T$ samples → $T$번 forward pass. LLM에선 비용 큼.
   - 단일 forward pass로 Laplace 대체 가능

**응용**:
- **RLHF calibration**: reward model uncertainty
- **Active learning for fine-tuning data selection**

**Open problem**: Full LLM posterior는 여전히 intractable. Component-wise Bayesian 근사가 현재 best approach.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. MC Dropout](./04-mc-dropout.md) | [📚 README로 돌아가기](../README.md) | [Ch6-01. GP 기반 BO 프레임워크 ▶](../ch6-bayesian-optimization/01-gp-bo-framework.md) |

</div>
