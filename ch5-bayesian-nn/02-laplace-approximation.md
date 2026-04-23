# 02. Laplace Approximation

## 🎯 핵심 질문

- Posterior 최빈값 $W^*$ 주변 **2차 Taylor** 전개로 $p(W|D) \approx \mathcal{N}(W^*, H^{-1})$이 어떻게 유도되는가?
- Hessian $H = -\nabla^2 \log p(W|D)|_{W^*}$가 **Fisher 정보**와 같음을 어떻게 증명?
- 현대 **Kronecker-factored Laplace**(KFAC, Ritter et al. 2018)가 어떻게 고차원을 다루는가?
- 학습된 NN에 **post-hoc**으로 uncertainty를 더하는 방법?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

Laplace는 **사후 학습**(post-hoc)에 적용 가능 — 이미 학습된 NN에 uncertainty를 "붙임". 재학습 불필요. BNN 중 **가장 computational-friendly**. 최근 `laplace-torch` 라이브러리로 대규모 NN(ResNet, BERT)에도 적용 가능. BvM(Ch1-05)의 자연스러운 계산적 실현.

---

## 📐 수학적 선행 조건

- [Ch1-05 Bernstein–von Mises](../ch1-bayesian-foundation/05-bernstein-von-mises.md): posterior asymptotic Gaussianity
- [Ch5-01 BNN](./01-bnn-formulation.md)
- [Mathematical Statistics Deep Dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive): Fisher 정보
- Taylor 전개, Hessian, Kronecker product

---

## 📖 직관적 이해

### Laplace의 기본 아이디어

$\log p(W|D)$를 **MAP 위치 $W^*$ 주변 2차 Taylor**:

$$\log p(W|D) \approx \log p(W^*|D) - \frac{1}{2}(W - W^*)^T H (W - W^*)$$

$\nabla\log p$가 $W^*$에서 0 (MAP), 2차 항만 남음.

Exponential:
$$p(W|D) \approx \exp\left(\log p(W^*|D)\right) \cdot \exp\left(-\frac{1}{2}(W - W^*)^T H (W - W^*)\right)$$

⇒ **$\mathcal{N}(W^*, H^{-1})$** Gaussian posterior.

### 왜 Hessian = Fisher?

Log-likelihood의 **negative Hessian** = Fisher 정보 (Ch1-05 정의 5.2):

$$H = -\nabla^2 \log p(D|W) - \nabla^2 \log p(W) \approx F(W^*) + \Sigma_0^{-1}$$

(prior가 $\mathcal{N}(0, \Sigma_0)$이면).

### 한계: $d \times d$ Hessian

$d = 10^6$이면 $10^{12}$ entries — 저장조차 불가능. **Kronecker factored** 근사로 해결.

### 요리 비유

"요리사들의 분포를 알아내려 하는데, 대신 **가장 잘하는 요리사 주변의 비슷한 요리사 원** 정도만 알자 — Gaussian 근사".

---

## ✏️ 엄밀한 정의

### 정의 2.1 — MAP Solution

$$W^* = \arg\max_W \log p(W|D) = \arg\max_W [\log p(D|W) + \log p(W)]$$

### 정의 2.2 — Laplace Approximation

$$p(W|D) \approx \mathcal{N}(W^*, H^{-1})$$

where $H = -\nabla^2 \log p(W|D)|_{W^*}$.

### 정의 2.3 — Gauss-Newton Approximation

Cross-entropy classification loss의 Hessian을 **Gauss-Newton**으로 근사:

$$H_{GN} = \sum_i J_i^T \Lambda_i J_i + \Sigma_0^{-1}$$

where $J_i = \nabla_W f_W(x_i)$ (Jacobian), $\Lambda_i$ = output-space Hessian of loss.

### 정의 2.4 — Kronecker-Factored Approximation (KFAC)

각 layer의 weight matrix $W_\ell$에 대해, layer별 Hessian block $H_\ell$을:

$$H_\ell \approx A_\ell \otimes B_\ell$$

where $A_\ell$ = input activation의 covariance, $B_\ell$ = gradient의 covariance.

이렇게 하면 저장이 $O(n_{in}^2 + n_{out}^2)$ vs full $O(n_{in}^2 n_{out}^2)$.

---

## 🔬 정리와 증명

### 정리 2.1 — Laplace 유도

**명제**: $\log p(W|D)$가 $W^*$에서 local max이고 $C^2$이면:

$$p(W|D) \approx \mathcal{N}(W^*, H^{-1})$$

with $H = -\nabla^2 \log p(W|D)|_{W^*}$.

**증명**: Ch1-05 정리 5.1 (BvM)의 Laplace 근사 버전. Taylor + exponential. $\square$

### 정리 2.2 — Hessian = Fisher + Prior precision

**명제**: $p(W) = \mathcal{N}(0, \Sigma_0)$ prior 하에서:

$$H = -\sum_i \nabla^2 \log p(y_i|x_i, W^*) + \Sigma_0^{-1}$$

첫 항이 empirical **Fisher 정보** (regression·classification 같이):

$$F_{emp}(W^*) = \sum_i \mathbb{E}_{p(y|x_i, W^*)}[\nabla\log p\cdot(\nabla\log p)^T]$$

**증명 개요**: 
- Fisher: $F = \mathbb{E}[-\nabla^2 \log p] = \mathbb{E}[(\nabla\log p)(\nabla\log p)^T]$ (information identity)
- Empirical version uses $y_i$ from data

$\square$

**귀결**: NN Hessian 계산 = Fisher 계산 (backprop Jacobian으로 가능).

### 정리 2.3 — Laplace Model Evidence

**명제**: Laplace approximation으로 model evidence:

$$\log p(D) \approx \log p(D|W^*) + \log p(W^*) + \frac{d}{2}\log(2\pi) - \frac{1}{2}\log|H|$$

**증명**:
$$p(D) = \int p(D|W)p(W)dW \approx p(D|W^*)p(W^*) \cdot (2\pi)^{d/2}|H|^{-1/2}$$

Gaussian integral. $\square$

**실전**: Model comparison에 Laplace evidence 사용. **BIC**는 이의 점근 근사 ($d\log N/2$).

### 정리 2.4 — Linearized Laplace Predictive

**명제**: Classification의 경우 Gauss-Newton Hessian + **NN linearization**으로:

$$f_W(x^*) \approx f_{W^*}(x^*) + J_{x^*}(W - W^*)$$

이면 predictive:

$$p(y^*|x^*, D) \approx \int p(y^*|f_{W^*}(x^*) + J_{x^*}(W - W^*)) \mathcal{N}(W; W^*, H^{-1}) dW$$

$W$에 대한 적분이 $J_{x^*}$를 통해 **1D** 적분으로 환원. 효율적.

**증명**: Linear-Gaussian marginalization. $\square$

**Laplace Bridge** (MacKay 1992): Dirichlet approximation 등으로 binary/categorical 정확도 개선.

### 정리 2.5 — KFAC의 근사 정합성

**명제**: Layer-wise weight $W_\ell \in \mathbb{R}^{n_{out} \times n_{in}}$의 Hessian block:

$$H_\ell = \mathbb{E}[a_\ell a_\ell^T \otimes g_\ell g_\ell^T]$$

($a_\ell$ = input, $g_\ell$ = output gradient). **Independence 가정** 하:

$$H_\ell \approx \mathbb{E}[a_\ell a_\ell^T] \otimes \mathbb{E}[g_\ell g_\ell^T] = A_\ell \otimes B_\ell$$

**증명**: Expectation of Kronecker product = Kronecker of expectations (IF independent).

$\square$

**실전 영향**: Martens & Grosse (2015) K-FAC — 2nd-order optimization. Ritter et al. (2018) 이 구조를 Laplace에 적용.

### 예시 — Last-Layer Laplace on MLP

학습된 MLP $f_W(x) = v^T \sigma(W_1 x + b_1) + b_2$에서 마지막 layer만:
- $w = (v, b_2)$ (수천 params)
- Prior $p(w) = \mathcal{N}(0, \tau^2 I)$
- Hessian $H = \sum_i \phi(x_i)\phi(x_i)^T + I/\tau^2$, $\phi(x) = [\sigma(W_1 x+b_1), 1]$
- Posterior $\mathcal{N}(\hat w, H^{-1})$

**Bayesian linear regression** with NN features. Tractable.

---

## 💻 PyTorch + laplace-torch

```python
# pip install laplace-torch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from laplace import Laplace

# 학습된 NN
model = nn.Sequential(nn.Linear(1, 50), nn.Tanh(),
                      nn.Linear(50, 50), nn.Tanh(),
                      nn.Linear(50, 1))

# Training data
x = torch.linspace(-3, 3, 50).unsqueeze(1)
y = torch.sin(x) + 0.1 * torch.randn_like(x)
loader = DataLoader(TensorDataset(x, y), batch_size=50)

# (사전에 point estimate 학습 — 생략)
optim = torch.optim.Adam(model.parameters(), lr=1e-2)
for epoch in range(2000):
    for xb, yb in loader:
        optim.zero_grad()
        loss = 0.5 * ((model(xb) - yb)**2).sum() + 1e-4 * sum((p**2).sum() for p in model.parameters())
        loss.backward(); optim.step()

# Laplace 근사: post-hoc
la = Laplace(model, 'regression', subset_of_weights='last_layer', 
             hessian_structure='full')
la.fit(loader)  # compute H, μ
la.optimize_prior_precision(method='marglik')  # optimize prior via marginal likelihood

# Predict with uncertainty
x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
f_mu, f_var = la(x_test)
f_mu = f_mu.detach().numpy().flatten()
f_std = f_var.sqrt().detach().numpy().flatten()

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.fill_between(x_test.numpy().flatten(), f_mu - 2*f_std, f_mu + 2*f_std, alpha=0.3,
                 label='±2σ (Laplace)')
plt.plot(x_test.numpy(), f_mu, 'b-', lw=2, label='Posterior mean')
plt.scatter(x.numpy(), y.numpy(), s=20, color='red', label='data')
plt.plot(x_test.numpy(), np.sin(x_test.numpy()), 'k--', alpha=0.5)
plt.legend(); plt.grid(alpha=0.3)
plt.title('Last-layer Laplace — uncertainty expands in extrapolation')
plt.savefig('laplace.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Post-Hoc Bayesian
학습은 표준 SGD, **inference 시** Bayesian 변환.

### BIC
$\text{BIC} = -2\log p(D|W^*) + d\log N$ — Laplace evidence의 점근 근사. Model selection.

### Kronecker-Factored Approximate Curvature
2nd-order optimization (K-FAC) 원래 용도, 이후 Laplace에 재사용.

### BvM Connection
정확한 posterior가 대규모에서 Gaussian이면 Laplace가 **asymptotically exact** (Ch1-05).

### Subspace Inference
High-dim Laplace 대신 top-$k$ eigendir만 Bayesian — 실전 compromise.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Unique local max $W^*$ | NN의 multiple minima → single Laplace is local |
| $\log p$가 quadratic 근처 | Highly skewed posterior에서 부정확 |
| Hessian positive-definite | Saddle point 또는 singular Hessian 문제 |
| $d$ 관리 가능 | Full Hessian $d = 10^6$ 불가 → KFAC/diagonal |
| Linearization 적절 | 강한 non-linearity는 linearized predictive 부정확 |

**실무 팁**:
- Full network Laplace는 last-layer 또는 KFAC
- Multi-modal multi-restart: each from different init → multiple Laplace Gaussians
- laplace-torch library: ResNet, BERT 지원

---

## 📌 핵심 정리

$$\boxed{p(W|D) \approx \mathcal{N}(W^*, H^{-1}), \quad H = -\nabla^2 \log p(W|D)|_{W^*}}$$

$$\boxed{H \approx F(W^*) + \Sigma_0^{-1}}$$

핵심:
- MAP 주변 2차 Taylor → Gaussian posterior
- Hessian = Fisher + prior precision
- Post-hoc 적용 가능
- KFAC로 고차원 확장

---

## 🤔 생각해볼 문제

**문제 1** (기초): 1D log-posterior $\log p(\theta|D) = -\frac{1}{2}(\theta - 1)^2 + \frac{1}{4}\theta^3$의 Laplace 근사는?

<details>
<summary>해설</summary>

$f(\theta) = -\frac{1}{2}(\theta-1)^2 + \frac{1}{4}\theta^3$.

$f'(\theta) = -(\theta - 1) + \frac{3}{4}\theta^2 = 0$.
$\frac{3}{4}\theta^2 - \theta + 1 = 0$.
$\theta = \frac{1 \pm \sqrt{1 - 3}}{3/2}$ → complex. No real root → no MAP.

**실례 수정**: $-\frac{1}{4}\theta^3$로 바꿈. 그러면 $-(\theta-1) - \frac{3}{4}\theta^2 = 0$, $\theta = ...$.

또는 단순 예: $\log p = -\frac{1}{2}(\theta-1)^2 - 0.1\theta^4$. MAP 찾고 $f''$ 계산.

$f'(\theta) = -(\theta - 1) - 0.4\theta^3 = 0$.
$\theta^*$ ≈ 0.85 (수치).
$f''(\theta) = -1 - 1.2\theta^{*2} \approx -1.87$.

Laplace: $\mathcal{N}(0.85, 1/1.87) = \mathcal{N}(0.85, 0.535)$.

True posterior (non-Gaussian due to $-\theta^4$)와 비교하면 근사 오차 있음.

</details>

**문제 2** (심화): "Last-layer Laplace"와 "Full Laplace"의 차이는?

<details>
<summary>해설</summary>

**Last-layer**: feature map $\phi(x)$ = NN의 penultimate layer output. Bayesian linear regression on $\phi(x)$. Weights 수백~수천.

**Full**: 전체 weight에 Laplace. $d = 10^6$에선 Hessian 저장 어려움 → KFAC 필수.

| | Last-layer | Full (KFAC) |
|---|---|---|
| Computation | Cheap (linear-Gaussian) | Moderate (KFAC) |
| Expressiveness | Fixed features → limited | Full uncertainty |
| Underfitting risk | 높음 (features가 OOD 파악 못 함) | 낮음 |
| Calibration | Often 괜찮음 | Better |

**실전**: Last-layer가 **good baseline** — cheap & often effective.

</details>

**문제 3** (AI 연결): ChatGPT 같은 fine-tuned LLM의 uncertainty를 Laplace로 추정하려면?

<details>
<summary>해설</summary>

**문제**: $10^{11}$ params, 학습된 LLM에 full Laplace 불가.

**현실적 조합**:
1. **LoRA Laplace** (Yang et al. 2023): LoRA adapter params($10^6$)에만 Laplace
2. **Last-layer Laplace**: classification head에만
3. **KFAC + layer selection**: 일부 layer만 Bayesian
4. **Linearized model** 전체: pretrained features + Bayesian regression head

**응용**:
- Hallucination detection (epistemic variance)
- Reliable confidence scoring
- Active learning for fine-tuning

최근 **LoRA Laplace for RLHF** 연구 활발 — 작은 adapter에 Bayesian이 안정성·calibration 개선.

Ch7-04 calibration과 연결.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. BNN의 수학적 정식화](./01-bnn-formulation.md) | [📚 README로 돌아가기](../README.md) | [03. Variational BNN — Bayes by Backprop ▶](./03-variational-bnn.md) |

</div>
