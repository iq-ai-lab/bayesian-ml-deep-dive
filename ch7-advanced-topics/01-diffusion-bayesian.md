# 01. Diffusion Model의 Bayesian 해석

## 🎯 핵심 질문

- Forward process $q(x_t|x_{t-1})$ = sequential Gaussian noise의 **hierarchical VAE 해석**은?
- Reverse $p_\theta(x_{t-1}|x_t)$ = posterior denoising의 VI 재정립?
- **DDPM ELBO**의 각 항은 무엇이고 $\|\epsilon - \epsilon_\theta\|^2$ 손실과 어떻게 일치하는가?
- Diffusion이 왜 **hierarchical VI의 가장 성공한 형태**인가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

Diffusion Model은 현재 **이미지 생성의 state-of-art**. 하지만 수학적 기반은 **hierarchical VAE = hierarchical VI**. 이 연결을 이해하면 DDPM·Score-SDE·Flow Matching·SGLD가 모두 **하나의 Bayesian 프레임워크**로 통합. Bayesian DL의 최신 prestige application.

SDE 레포 Ch6과 교차 — 여기선 **Bayesian·VI 관점**에 집중.

---

## 📐 수학적 선행 조건

- [Ch3-01 VAE](../ch3-vae-modern-variational/01-vae-derivation.md)
- [Ch5-05 SGLD](../ch5-bayesian-nn/05-swag-sgld.md): Langevin
- [SDE Deep Dive Ch6](https://github.com/iq-ai-lab/sde-deep-dive): Anderson reverse SDE
- DDPM의 forward/reverse process

---

## 📖 직관적 이해

### Forward Process

$T$-step Markov chain:
$$q(x_t | x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

$x_0$ (data) → $x_T \sim \mathcal{N}(0, I)$ (noise). **Fixed**, no parameter.

### Reverse Process

$$p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

Learnable — NN $\epsilon_\theta$가 core.

### ELBO

$$\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log \frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

Telescoping expansion:
$$-\mathcal{L} = \underbrace{\text{KL}(q(x_T|x_0)\|p(x_T))}_{L_T} + \sum_{t=1}^T \underbrace{\text{KL}(q(x_{t-1}|x_t, x_0)\|p_\theta(x_{t-1}|x_t))}_{L_{t-1}} - \underbrace{\log p_\theta(x_0|x_1)}_{L_0}$$

**$T$개의 KL terms** — $T$-level hierarchical VAE.

### 요리 비유

- VAE: "재료 → 한 번에 레시피 만들기"
- DDPM: "재료 → **조금씩 노이즈** (forward) → 조금씩 복원 (reverse)". $T$번 공정.

각 작은 단계가 **쉬운 local 문제** → 큰 step보다 학습 쉬움.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Forward Process

$$q(x_{1:T}|x_0) = \prod_{t=1}^T q(x_t|x_{t-1})$$

$$q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$$

Marginal (Ch2-05 reparam):
$$q(x_t|x_0) = \mathcal{N}(\sqrt{\bar\alpha_t}x_0, (1-\bar\alpha_t)I)$$

$\bar\alpha_t = \prod_{s \leq t}(1-\beta_s)$.

### 정의 1.2 — Reverse Process

$$p_\theta(x_{0:T}) = p(x_T)\prod_{t=1}^T p_\theta(x_{t-1}|x_t)$$

$p(x_T) = \mathcal{N}(0, I)$.

### 정의 1.3 — DDPM Loss

$$\mathcal{L}_{\text{simple}} = \mathbb{E}_{t, x_0, \epsilon}\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right], \quad x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$$

### 정의 1.4 — Score Network

$$s_\theta(x_t, t) \approx \nabla_{x_t}\log p_t(x_t)$$

Relationship: $\epsilon_\theta \cdot$ const $= s_\theta$.

---

## 🔬 정리와 증명

### 정리 1.1 — DDPM ELBO Decomposition

**명제**:

$$-\log p_\theta(x_0) \leq \underbrace{\mathbb{E}_q[L_T]}_{\text{prior KL}} + \sum_{t > 1}\mathbb{E}_q[L_{t-1}] + \mathbb{E}_q[L_0]$$

where:
- $L_T = \text{KL}(q(x_T|x_0)\|p(x_T))$ — 고정, negligible
- $L_{t-1} = \text{KL}(q(x_{t-1}|x_t, x_0)\|p_\theta(x_{t-1}|x_t))$ — denoising KL
- $L_0 = -\log p_\theta(x_0|x_1)$ — reconstruction

**증명**:

$$\log p_\theta(x_0) = \log \int p_\theta(x_{0:T})dx_{1:T}$$

$q(x_{1:T}|x_0)$로 importance weighting:
$$\geq \mathbb{E}_q\left[\log\frac{p_\theta(x_{0:T})}{q(x_{1:T}|x_0)}\right]$$

Telescoping:
$$= \mathbb{E}_q[\log p(x_T)] - \text{KL}(q\|\cdot)\ldots$$

Bayes rule on $q(x_{t-1}|x_t, x_0) = q(x_t|x_{t-1})q(x_{t-1}|x_0)/q(x_t|x_0)$ (Markov + forward marginal).

상세 유도는 Ho et al. 2020 Appendix A. $\square$

### 정리 1.2 — $q(x_{t-1}|x_t, x_0)$의 Gaussian Closed Form

**명제**: 
$$q(x_{t-1}|x_t, x_0) = \mathcal{N}(\tilde\mu_t(x_t, x_0), \tilde\beta_t I)$$

with $\tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1 - \bar\alpha_t}x_0 + \frac{\sqrt{1-\beta_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$.

**증명**: Gaussian-Gaussian conjugate (Ch1-03). $\square$

### 정리 1.3 — DDPM Loss = Simple MSE

**명제** (Ho et al. 2020): $p_\theta(x_{t-1}|x_t) = \mathcal{N}(\mu_\theta(x_t, t), \beta_t I)$ with:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{1-\beta_t}}\left(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon_\theta(x_t, t)\right)$$

이면 $L_{t-1}$가 **$\|\epsilon - \epsilon_\theta\|^2$**로 단순화 (up to scale).

**증명 스케치**:

$\tilde\mu_t - \mu_\theta$를 $\epsilon$ form으로 재표현. Gaussian KL이 $\|\tilde\mu - \mu_\theta\|^2 / \beta_t$ 형태. 그러면 $\|\epsilon - \epsilon_\theta\|^2$와 선형 관계. $\square$

**귀결**: 이론적으로 **weighted sum of $t$-dependent terms**, but Ho et al.은 **상수 weight** ($L_{\text{simple}}$)가 실전 성능 더 좋다고 발견.

### 정리 1.4 — DDPM ↔ Score Matching 동치성

**명제** (Vincent 2011; Ho et al. 2020):

$$\|\epsilon - \epsilon_\theta\|^2 \equiv \|s - s_\theta\|^2 \cdot (1-\bar\alpha_t)$$

where $s_\theta \propto -\epsilon_\theta / \sqrt{1-\bar\alpha_t}$.

즉 $\epsilon$-prediction과 **denoising score matching**이 수학적으로 같은 학습.

**증명**: Tweedie 공식 (SDE 레포 Ch6-02) 또는 Bayes rule on Gaussian. $\square$

### 정리 1.5 — Hierarchical VAE Limit

**명제**: DDPM = **$T$-layer hierarchical VAE** with:
- Latent hierarchy $z_1 = x_1, \ldots, z_T = x_T$
- Fixed encoder (forward process)
- Learnable decoder (reverse process)
- Shared NN parameters across levels (weight sharing by $t$)

$T \to \infty$에서 → **continuous-time SDE** (Song et al. 2021).

---

## 💻 PyTorch 구현 (Simplified DDPM)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ────────────────────────────────────────────────
# Simplified DDPM on 1D/2D toy data
# ────────────────────────────────────────────────
class SimpleScoreNet(nn.Module):
    def __init__(self, dim=2, hidden=64):
        super().__init__()
        self.dim = dim
        # Input: [x, t_embedding]
        self.net = nn.Sequential(
            nn.Linear(dim + 16, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, dim)
        )
    def time_embed(self, t, emb_dim=16):
        half = emb_dim // 2
        freqs = torch.exp(-np.log(10000) * torch.arange(half)/half)
        args = t.unsqueeze(1) * freqs.unsqueeze(0)
        return torch.cat([args.sin(), args.cos()], dim=1)
    def forward(self, x, t):
        t_emb = self.time_embed(t.float())
        return self.net(torch.cat([x, t_emb], dim=1))

# Diffusion schedule
T = 100
betas = torch.linspace(1e-4, 0.02, T)
alphas = 1 - betas
alphas_bar = torch.cumprod(alphas, dim=0)

# Data: 2D two-moons
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=3000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

model = SimpleScoreNet(dim=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
for epoch in range(2000):
    idx = torch.randperm(len(X))[:256]
    x0 = X[idx]
    t = torch.randint(0, T, (len(x0),))
    alpha_bar_t = alphas_bar[t].unsqueeze(1)
    eps = torch.randn_like(x0)
    xt = torch.sqrt(alpha_bar_t)*x0 + torch.sqrt(1 - alpha_bar_t)*eps
    
    eps_pred = model(xt, t)
    loss = F.mse_loss(eps_pred, eps)   # L_simple
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 500 == 0:
        print(f"ep {epoch}: loss = {loss.item():.4f}")

# Sampling (ancestral)
@torch.no_grad()
def sample(n=1000):
    x = torch.randn(n, 2)
    for t in range(T - 1, -1, -1):
        t_tensor = torch.full((n,), t, dtype=torch.long)
        beta_t = betas[t]
        alpha_t = alphas[t]
        alpha_bar_t = alphas_bar[t]
        eps_pred = model(x, t_tensor)
        mean = (1/torch.sqrt(alpha_t)) * (x - beta_t/torch.sqrt(1-alpha_bar_t)*eps_pred)
        if t > 0:
            noise = torch.randn_like(x)
            x = mean + torch.sqrt(beta_t)*noise
        else:
            x = mean
    return x

samples = sample(1000)

import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(X[:, 0], X[:, 1], s=2, alpha=0.5); axes[0].set_title('Data')
axes[1].scatter(samples[:, 0], samples[:, 1], s=2, alpha=0.5); axes[1].set_title('DDPM samples')
for ax in axes: ax.set_xlim(-2, 3); ax.set_ylim(-1.5, 1.5); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('ddpm_toy.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Score-Based SDE
DDPM = discrete VP-SDE. Continuous limit → Anderson reverse SDE (SDE 레포 Ch6).

### Langevin Sampling
Annealed Langevin (NCSN, Song & Ermon 2019) = 다른 형태 score-based.

### ELBO vs $L_{\text{simple}}$
Ho et al.: $L_{\text{simple}}$ (unweighted) > ELBO (weighted by VLB). Theory-practice gap.

### Stable Diffusion
Latent Diffusion (Rombach 2022) = VAE + Diffusion 조합. $T = 1000$에서 $T \sim 50$로 reduced.

### Classifier-Free Guidance
Conditional vs unconditional score의 interpolation — Ho & Salimans 2022.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Gaussian forward process | Continuous data에만 직접 (discrete은 custom) |
| $T = 1000$ typical | Sampling 느림 → DDIM, consistency models |
| $\beta_t$ schedule 고정 | Learnable schedule 연구 |
| $\|\epsilon - \epsilon_\theta\|^2$ proxy | True ELBO와 다름, 학습 품질·likelihood 간 trade |

---

## 📌 핵심 정리

$$\boxed{\text{DDPM} = T\text{-step hierarchical VAE + 가중 DSM}}$$

$$\boxed{\mathcal{L}_{\text{simple}} = \mathbb{E}[\|\epsilon - \epsilon_\theta(x_t, t)\|^2]}$$

핵심:
- Forward = fixed Markov Gaussian chain
- Reverse = learnable → posterior denoising
- ELBO = $T$ KL terms + reconstruction
- Score/epsilon-prediction ≡ DSM
- SDE 레포 Ch6과 교차 (continuous limit)

---

## 🤔 생각해볼 문제

**문제 1** (기초): $T \to \infty$에서 DDPM은 무엇과 같아지는가?

<details>
<summary>해설</summary>

**Continuous-time SDE** (Song et al. 2021):

Forward: $dx = -\frac{1}{2}\beta(t)x\,dt + \sqrt{\beta(t)}\,dB_t$ (VP-SDE)

Reverse (Anderson): $dx = [-\frac{1}{2}\beta x - \beta \nabla\log p_t]dt + \sqrt\beta\,d\bar B_t$

**Score**: $s_\theta \approx \nabla\log p_t$ 학습.

$T$-discrete DDPM이 이 SDE의 **Euler-Maruyama 이산화** — SDE 레포 Ch6에서 엄밀히 증명.

**이점**:
- Sampling flexibility (any integrator)
- Probability flow ODE로 deterministic sampling (DDIM)
- Exact likelihood computation

</details>

**문제 2** (심화): $\|\epsilon - \epsilon_\theta\|^2$ (simple) vs ELBO-weighted ($\beta_t$ scaling) — 어느 것이 더 나은 이유?

<details>
<summary>해설</summary>

**Theoretical**: ELBO가 log-likelihood의 tight bound → 학습해야 하는 것.

**Empirical** (Ho et al. 2020): $L_{\text{simple}}$ (unweighted) → 이미지 **품질** 개선. ELBO-weighted는 log-likelihood에 더 집중.

**이유** (대략적 해석):
- Early $t$ (large noise): 덜 중요 (detail 없음)
- Late $t$ (small noise): 이미지 detail → 중요
- $L_{\text{simple}}$이 late $t$에 **상대적으로 더 많은 weight** → perceptual quality 강조

**Tradeoff**:
- Better FID: $L_{\text{simple}}$
- Better log-likelihood: ELBO-weighted
- 최근 Salimans-Ho 2022 "Progressive Distillation" — 두 objective 간 균형

</details>

**문제 3** (AI 연결): Stable Diffusion의 **latent diffusion**은 VAE와 DDPM을 어떻게 결합?

<details>
<summary>해설</summary>

**2-stage**:
1. **VAE**: $x$ (이미지, $512\times 512 \times 3$) → $z$ (latent, $64\times 64 \times 4$). 대폭 dimension 감소.
2. **DDPM** in latent space: $z_0 \to z_t$ with Gaussian noise, reverse 학습.

**이점**:
- VAE reconstruction은 perceptually 거의 완벽 (imperceptible loss)
- DDPM in $z$-space: **작은 dim** → 훨씬 빠른 training/inference
- **Conditional**: text embedding을 cross-attention로 주입 (CLIP)

**Bayesian 관점**:
- VAE: $q_\phi(z|x)$, $p_\theta(x|z)$ — Ch3-01
- DDPM on $z$: hierarchical VI on latent space
- **합**: 4-layer hierarchical VAE (VAE encoder → VAE decoder → DDPM) with Gaussian priors

Stable Diffusion = **두 Bayesian methods의 통합**. 현대 생성 모델의 가장 성공한 아키텍처.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch6-04 BO의 실전 확장](../ch6-bayesian-optimization/04-bo-extensions.md) | [📚 README로 돌아가기](../README.md) | [02. Probabilistic Programming 언어 ▶](./02-probabilistic-programming.md) |

</div>
