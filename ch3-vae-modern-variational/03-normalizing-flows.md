# 03. Normalizing Flows

## 🎯 핵심 질문

- 역가능 변환 $z_K = f_K\circ\cdots\circ f_1(z_0)$에서 밀도가 어떻게 바뀌는가?
- **변수변환 공식** $\log p(z_K) = \log p(z_0) - \sum\log|\det J_{f_k}|$는 왜?
- Planar·Radial·Real NVP·MAF 같은 flow family는 Jacobian을 어떻게 효율적으로 계산?
- Flow가 VAE의 **Gaussian encoder 제약**을 어떻게 완화하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**표현력 있는 posterior 근사**. VAE의 mean-field Gaussian은 단순 — 복잡한 posterior(multimodal, skewed)를 못 잡음. Flow는 **어떤 분포**로도 변환 가능한 유연한 family. **Neural ODE**, **Continuous Normalizing Flow**, **FFJORD**, **Glow**(이미지 생성) 등의 수학적 기반. **Likelihood 정확 계산** 가능 — diffusion/VAE가 ELBO만 주는 것과 달리 exact log $p(x)$.

---

## 📐 수학적 선행 조건

- [Ch3-01 VAE](./01-vae-derivation.md)
- 다변수 미분: Jacobian, determinant
- 변수변환 공식 (change of variables)
- Linear algebra: triangular matrix의 det

---

## 📖 직관적 이해

### Flow의 기본 아이디어

**간단한 base 분포** $z_0 \sim p_0$ (Gaussian 등)을 **연속적 변환**으로 복잡한 분포로:

$$z_0 \xrightarrow{f_1} z_1 \xrightarrow{f_2} z_2 \xrightarrow{f_3} \cdots \xrightarrow{f_K} z_K$$

각 $f_k$가 **역가능(invertible)**이면 밀도를 정확히 추적 가능.

### 변수변환 공식

$y = f(x)$, $f$ invertible:

$$p_Y(y) = p_X(f^{-1}(y)) \cdot |\det J_{f^{-1}}(y)| = p_X(x)/|\det J_f(x)|$$

로그:
$$\log p_Y(y) = \log p_X(x) - \log|\det J_f(x)|$$

Composition에서 Jacobian이 **곱해짐** → log가 **더해짐**.

### 왜 "normalizing"인가

복잡한 $p(x)$를 "정규화해서" 간단한 Gaussian으로 **역방향** 관점: $x = f^{-1}(z_0)$로 $x$를 Gaussian $z_0$로 "normalize". 또는 순방향: Gaussian에서 시작해 복잡한 target을 만듦.

### 요리 비유

- Base: 밋밋한 물 (Gaussian)
- Flow: "**순차적 요리 단계**(볶기 → 끓이기 → 간하기)" 각각 역가능
- 결과: 복잡한 요리 (target 분포)
- Jacobian = "각 단계에서 부피 변화"

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Normalizing Flow

역가능 미분 가능 함수 $f: \mathbb{R}^d \to \mathbb{R}^d$ 사이에서:

$$z_0 \sim p_0, \quad z_k = f_k(z_{k-1}), \quad k = 1, \ldots, K$$

각 $f_k$는 $C^1$, invertible.

### 정의 3.2 — Log Density Transformation

$$\log p_K(z_K) = \log p_0(z_0) - \sum_{k=1}^K \log|\det J_{f_k}(z_{k-1})|$$

이 공식으로 final 분포의 log-density를 **정확히** 계산.

### 정의 3.3 — Tractable Flow Families

효율적 학습을 위한 조건:

1. $f_k$ **역가능**
2. $|\det J_{f_k}|$ **효율적 계산** (일반적으로 $O(d^3)$ → Flow design으로 $O(d)$)
3. $f_k$가 **표현력** 있음

---

## 🔬 정리와 증명

### 정리 3.1 — Change of Variables Formula

**명제**: $Y = f(X)$, $f: \mathbb{R}^d \to \mathbb{R}^d$ diffeomorphism이면:

$$p_Y(y) = p_X(f^{-1}(y))|\det J_{f^{-1}}(y)| = \frac{p_X(x)}{|\det J_f(x)|}\bigg|_{x = f^{-1}(y)}$$

**증명**: 측도 변환. For any measurable $A$:
$$P(Y \in A) = P(X \in f^{-1}(A)) = \int_{f^{-1}(A)}p_X(x)dx$$

$y = f(x), x = f^{-1}(y), dx = |\det J_{f^{-1}}|dy$:

$$= \int_A p_X(f^{-1}(y))|\det J_{f^{-1}}(y)|dy$$

$\square$

### 정리 3.2 — Composition의 Jacobian

**명제**: $f = f_K\circ\cdots\circ f_1$이면:

$$\log|\det J_f(x)| = \sum_{k=1}^K \log|\det J_{f_k}(z_{k-1})|$$

**증명**: Chain rule: $J_f = J_{f_K}\cdot J_{f_{K-1}}\cdots J_{f_1}$. $\det$의 multiplicativity:
$$\det J_f = \prod_k \det J_{f_k}$$
$\log$ 취함. $\square$

### 정리 3.3 — Planar Flow (Rezende & Mohamed 2015)

**명제**: $f(z) = z + u\cdot h(w^T z + b)$ where $h$ scalar. Jacobian:

$$J_f = I + u\cdot h'(w^T z + b)w^T$$

**Matrix determinant lemma**:
$$\det(I + ab^T) = 1 + a^T b$$

$$|\det J_f| = |1 + h'(w^T z + b)u^T w|$$

**증명**: Matrix determinant lemma 직접 적용. $\square$

**효율성**: $O(d)$ (rank-1 update). 단일 flow의 표현력은 제한 — 많은 layer 필요.

### 정리 3.4 — Real NVP (Dinh, Sohl-Dickstein, Bengio 2017)

**명제**: Coupling layer — $z = (z_1, z_2)$, $z_1 \in \mathbb{R}^{d_1}, z_2 \in \mathbb{R}^{d_2}$:

$$\begin{aligned}y_1 &= z_1\\ y_2 &= z_2\odot \exp(s(z_1)) + t(z_1)\end{aligned}$$

Jacobian is lower triangular:

$$J = \begin{pmatrix}I & 0\\ \partial y_2/\partial z_1 & \text{diag}(\exp(s(z_1)))\end{pmatrix}$$

$$\log|\det J| = \sum_i s_i(z_1)$$

**역변환**: 
$$z_2 = (y_2 - t(y_1))\odot\exp(-s(y_1))$$ — exact & explicit.

**증명**: Triangular matrix의 det = diagonal product. $\square$

**특징**: $s, t$는 임의 NN (제약 없음). **매우 유연**. 연속된 coupling layer로 permutation 섞어서 $(z_1, z_2)$ 역할 교환.

### 정리 3.5 — Masked Autoregressive Flow (MAF, Papamakarios 2017)

**명제**: Autoregressive variant:

$$y_i = (z_i - \mu_i(y_{1:i-1}))\exp(-\alpha_i(y_{1:i-1}))$$

Jacobian triangular ⇒ $\log|\det J| = -\sum_i \alpha_i$.

**Density evaluation**: $O(d)$ per step.  
**Sampling**: $O(d)$ sequentially (autoregressive).

Inverse autoregressive flow(IAF, Kingma 2016): sampling fast, density slow.

### 정리 3.6 — Glow (Kingma & Dhariwal 2018)

- Invertible 1×1 convolution (LU decomposition → fast det)
- Affine coupling (like Real NVP)
- Actnorm (normalization)

이미지 생성에서 Real NVP보다 월등 — FID 수백점 → 수십점.

### 예시 — 1D Flow

$z_0 \sim \mathcal{N}(0, 1)$, $z_1 = \tanh(z_0)$ (but $\tanh$ 역가능 bijective → OK on $(-1, 1)$).

$p_1(z_1) = p_0(\text{atanh}(z_1))/|(1 - z_1^2)|$

2D flow를 통해 Gaussian에서 moon-shape, checkerboard 등 학습 가능.

---

## 💻 NumPy + PyTorch 구현

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# ────────────────────────────────────────────────
# Real NVP — 2D toy flow
# ────────────────────────────────────────────────
class CouplingLayer(nn.Module):
    def __init__(self, dim, hidden=64, flip=False):
        super().__init__()
        self.flip = flip
        self.d = dim // 2
        self.net = nn.Sequential(
            nn.Linear(self.d, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 2 * (dim - self.d))
        )

    def forward(self, x):
        if self.flip: x = x.flip(dims=[-1])
        x1, x2 = x[:, :self.d], x[:, self.d:]
        st = self.net(x1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)           # stabilize
        y2 = x2 * torch.exp(s) + t
        y = torch.cat([x1, y2], dim=-1)
        if self.flip: y = y.flip(dims=[-1])
        log_det = s.sum(dim=-1)
        return y, log_det

    def inverse(self, y):
        if self.flip: y = y.flip(dims=[-1])
        y1, y2 = y[:, :self.d], y[:, self.d:]
        st = self.net(y1)
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        x2 = (y2 - t) * torch.exp(-s)
        x = torch.cat([y1, x2], dim=-1)
        if self.flip: x = x.flip(dims=[-1])
        return x

class RealNVP(nn.Module):
    def __init__(self, dim=2, n_layers=8):
        super().__init__()
        self.layers = nn.ModuleList([
            CouplingLayer(dim, flip=(i % 2 == 1)) for i in range(n_layers)
        ])
        self.base = torch.distributions.Normal(0, 1)

    def log_prob(self, x):
        log_det_sum = 0
        z = x
        for layer in self.layers:
            z, log_det = layer(z)
            log_det_sum = log_det_sum + log_det
        # z ~ base (after training)
        log_p_z = self.base.log_prob(z).sum(dim=-1)
        return log_p_z + log_det_sum

    def sample(self, n):
        z = self.base.sample((n, 2))
        for layer in reversed(self.layers):
            z = layer.inverse(z)
        return z

# Train on two-moons
from sklearn.datasets import make_moons
X, _ = make_moons(n_samples=2000, noise=0.05)
X = torch.tensor(X, dtype=torch.float32)

model = RealNVP(dim=2)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(3000):
    idx = torch.randperm(len(X))[:256]
    x = X[idx]
    loss = -model.log_prob(x).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if epoch % 500 == 0:
        print(f"epoch {epoch}: NLL = {loss.item():.3f}")

# Sample and plot
with torch.no_grad():
    samples = model.sample(2000).numpy()

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].scatter(X[:, 0], X[:, 1], s=3, alpha=0.5)
axes[0].set_title('Data (two-moons)')
axes[1].scatter(samples[:, 0], samples[:, 1], s=3, alpha=0.5)
axes[1].set_title('Flow samples')
for ax in axes: ax.set_xlim(-2, 3); ax.set_ylim(-1.5, 1.5); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('flow_twomoons.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Flow-Based VAE
Flow가 VAE posterior: $q_\phi(z|x) = $ Gaussian 후 flow → 복잡한 posterior. 더 tight ELBO.

### Glow
이미지 생성. Flow가 VAE/GAN 대비 **exact likelihood** 제공 — anomaly detection에 유리.

### Continuous Normalizing Flow (FFJORD, Neural ODE)
연속시간 flow $dz/dt = f(z, t)$. Chen et al. 2018. **Infinite depth** equivalent.

### Probability Flow ODE (Diffusion ↔ Flow)
SDE 레포 Ch7-01. Diffusion의 deterministic sampling = continuous flow.

### Flow Matching (Lipman 2023)
SM의 flow 버전 — target score field 대신 target velocity field 학습.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $f_k$ bijective | 차원 변경 불가 ($\mathbb{R}^d \to \mathbb{R}^d$) |
| Efficient Jacobian | 특수 구조 설계 필요 (coupling/autoregressive) |
| Differentiable $f$ | Discrete data 직접 불가 (dequantization trick 필요) |
| Expressive depth | 수십~수백 layer 필요 (VAE·Diffusion 대비 parameter efficiency 낮음) |

**실무 팁**: **이미지 생성**은 현대엔 Diffusion > Flow. **Likelihood 정확도**가 중요하면 Flow. **Small-dim posterior** 근사(VI 개선)엔 Flow 우수.

---

## 📌 핵심 정리

$$\boxed{\log p(z_K) = \log p(z_0) - \sum_{k=1}^K\log|\det J_{f_k}(z_{k-1})|}$$

주요 flow family:
- **Planar**: rank-1 update ($O(d)$ det)
- **Real NVP**: coupling ($d/2$ → $d/2$, triangular Jacobian)
- **MAF/IAF**: autoregressive
- **Glow**: 1×1 conv + affine coupling

VAE 대비 장점: **exact likelihood** + 유연한 posterior.
단점: 차원 제약, deep network 필요.

---

## 🤔 생각해볼 문제

**문제 1** (기초): $y = \sigma(x) = 1/(1+e^{-x})$의 Jacobian과 변환된 density는?

<details>
<summary>해설</summary>

$dy/dx = \sigma(x)(1-\sigma(x))$.

$p_Y(y) = p_X(\sigma^{-1}(y))/[\sigma(x)(1-\sigma(x))] = p_X(\text{logit}(y))/(y(1-y))$.

Used in normalizing flow for bounded outputs. 예: $Y \in (0, 1)$를 $X \in \mathbb{R}$로 변환한 뒤 flow 적용.

</details>

**문제 2** (심화): Real NVP의 $s(z_1) = 0, t(z_1) = 0$이면 $y = z$ (identity). 왜 그래도 $s, t$가 복잡하게 학습되나?

<details>
<summary>해설</summary>

Composition of coupling layers with **permutation**로 $(z_1, z_2)$ 역할 번갈아 → 모든 차원에 영향.

각 layer 자체는 일부만 변환하지만 **충분한 layer 수** (예: 8~32)로 조합 → 복잡한 분포 표현.

$s \to 0, t \to 0$은 initial 상태 (identity로 시작하는 것이 안정). 학습하면서 점차 유의미한 nonlinearity 획득.

Real NVP 한 layer의 표현력 ≠ 많은 layer의 집합의 표현력.

</details>

**문제 3** (AI 연결): Flow와 Diffusion의 **trade-off**는?

<details>
<summary>해설</summary>

| | Flow | Diffusion |
|---|---|---|
| Likelihood | **Exact** | ELBO only (DDPM) / exact via PF-ODE |
| Training | Max likelihood 직접 | DSM / VLB |
| Architecture | 특수 구조 (bijective) | **자유로운 NN** |
| Sampling | One-shot | Iterative (T steps) |
| Quality | 제한적 (FID higher) | **State-of-art** |
| Parameter efficiency | Low (deep) | Medium |

현대 흐름: **이미지 생성은 Diffusion 우세**. Flow는 **특정 niche** — likelihood 필요, 작은 모델, tabular data.

**Flow Matching** (Lipman 2023, Ch7-01 연관)은 Flow와 Diffusion의 이점 통합 시도.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. VAE의 변종](./02-vae-variants.md) | [📚 README로 돌아가기](../README.md) | [04. Amortized Inference ▶](./04-amortized-inference.md) |

</div>
