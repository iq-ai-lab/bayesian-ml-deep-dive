# 04. MC Dropout = Approximate VI (Gal & Ghahramani 2016)

## 🎯 핵심 질문

- Dropout rate $p$의 NN이 어떻게 **Bernoulli variational posterior**와 동치인가?
- Test-time에 dropout을 유지하고 $T$번 forward pass로 MC predictive를 얻는 원리?
- 왜 MC Dropout이 **"공짜 BNN"**이라 불리는가?
- Dropout이 어떤 prior에 해당하고 어떤 한계를 갖는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

MC Dropout은 **실전 BNN의 가장 간단하고 효과적인 방법** — 표준 dropout 있는 NN을 **그대로 재사용**하여 uncertainty 얻음. Gal & Ghahramani (2016)의 PhD 논문이 학계·산업계에 BNN을 친숙하게 만듦. Medical imaging, autonomous driving의 uncertainty estimation이 대부분 MC Dropout 기반. 구현 20줄 이내.

---

## 📐 수학적 선행 조건

- [Ch2 VI 전체](../ch2-variational-inference/01-vi-idea-elbo.md)
- [Ch5-01~03](./01-bnn-formulation.md): BNN, Laplace, BBB
- Dropout 기본 (Srivastava et al. 2014)

---

## 📖 직관적 이해

### Dropout의 Bayesian 재해석

표준 dropout: 학습 시 각 neuron을 $p$ 확률로 끔, test 시 weight scaling.

**MC Dropout**: test 시에도 dropout 유지, $T$번 forward pass → **MC sample of predictive**.

```
For test x*:
   predictions = []
   For t = 1..T:
      y_t = f(x*, with_dropout=True)  # each run different mask
      predictions.append(y_t)
   Return mean, variance of predictions
```

### Variational 관점

Weight matrix $W$에 Bernoulli mask:
$$W_{\text{effective}} = W \cdot \text{Bernoulli}(1-p)$$

이것이 variational posterior $q(W) = \prod_{ij}[p\cdot\delta_0 + (1-p)\cdot\delta_{M_{ij}}]$와 동치.

### 왜 "Free BNN"

- 별도 학습 안 함 (이미 dropout으로 학습된 모델 사용)
- 추가 파라미터 없음 ($\mu$, $\sigma$ 별도 저장 불필요 — BBB와 다름)
- 구현 간단 (`model.train()` mode at inference)

### 요리 비유

"요리사들이 **매번 랜덤하게 몇 명 쉼** — 남은 요리사가 협력. 여러 번 돌리면 요리의 **다양성 = uncertainty**".

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Dropout as Variational Approximation

Weight matrix $W_\ell \in \mathbb{R}^{n_{out} \times n_{in}}$. Dropout with rate $p$:

$$W_\ell^{(t)} = M_\ell \cdot \text{diag}(z^{(t)}), \quad z_i^{(t)} \sim \text{Bernoulli}(1 - p)$$

Variational posterior:
$$q(W_\ell) = M_\ell \cdot \text{diag}(z), \quad z \sim \text{Bernoulli}(1-p)^{n_{in}}$$

$M_\ell$ = trained mean weight (MAP-like).

### 정의 4.2 — MC Predictive

$$p(y^*|x^*, D) \approx \frac{1}{T}\sum_{t=1}^T p(y^*|x^*, W^{(t)})$$

$W^{(t)}$ = $t$번째 dropout masking.

### 정의 4.3 — Predictive Variance

**Regression**:
$$\text{Var}[y^*|x^*] \approx \sigma_n^2 + \frac{1}{T}\sum_t f_{W^{(t)}}(x^*)^2 - \left(\frac{1}{T}\sum_t f_{W^{(t)}}(x^*)\right)^2$$

첫 항 = aleatoric, 둘째 = epistemic (sample variance).

---

## 🔬 정리와 증명

### 정리 4.1 — Dropout = Variational Bernoulli Posterior

**명제** (Gal & Ghahramani 2016): $L$-layer NN with dropout rate $p_\ell$ 각 layer에서 ELBO minimization이:

$$\mathcal{L}_{\text{MC-dropout}}(M) = -\frac{1}{N}\sum_i \mathbb{E}_{q(W)}[\log p(y_i|x_i, W)] + \frac{1-p}{2N}\sum_\ell \|M_\ell\|^2$$

와 같음. 이것이 **standard dropout training objective** (cross-entropy + L2 reg).

**증명 개요** (informal):

1. ELBO = $\mathbb{E}_q[\log p(D|W)] - \text{KL}(q\|p_{prior})$
2. Bernoulli posterior의 KL w.r.t. Gaussian prior → $\|M\|^2$ 항으로 approximable
3. Mini-batch + reparam으로 cross-entropy + L2 형태

상세는 Gal 2016 thesis Appendix. $\square$

**귀결**: **standard dropout NN**은 이미 variational BNN → test-time에 dropout 유지하면 valid posterior sampling.

### 정리 4.2 — Predictive는 Correct Marginalization

**명제**: MC predictive가 true Bayesian predictive로 수렴:

$$\frac{1}{T}\sum_t p(y^*|x^*, W^{(t)}) \xrightarrow{T\to\infty} \mathbb{E}_{q(W)}[p(y^*|x^*, W)]$$

**증명**: LLN. $\square$

Family 제약 (Bernoulli mask) 때문에 **true posterior와는 차이** — 여전히 근사.

### 정리 4.3 — Dropout Rate와 Uncertainty 관계

**명제**: Dropout rate $p$ 높을수록 **predictive variance 증가**:
- $p = 0$: variance = 0 (deterministic)
- $p = 0.5$: maximum uncertainty
- $p$ 높음: 각 forward pass 더 다양

**실전**: $p = 0.1 \sim 0.5$가 보통. Too small → under-estimate uncertainty, too large → degraded accuracy.

### 정리 4.4 — Concrete Dropout (Gal et al. 2017)

**명제**: 각 layer의 $p_\ell$을 **학습 가능**하게 하는 confident Dropout:

$$p_\ell = \text{learnable}, \quad q_\ell(w) = p_\ell \delta_0 + (1-p_\ell)\delta_{M_\ell}$$

ELBO gradient on $p_\ell$ → data-driven dropout rate.

**Concrete (Gumbel-Softmax)** relaxation으로 differentiable.

**귀결**: Principled per-layer uncertainty calibration.

---

## 💻 PyTorch 구현 (놀라울 정도로 간단)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

class DropoutNN(nn.Module):
    def __init__(self, hidden=50, dropout=0.1):
        super().__init__()
        self.l1 = nn.Linear(1, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.l3 = nn.Linear(hidden, 1)
        self.dropout = dropout

    def forward(self, x):
        h = F.dropout(F.relu(self.l1(x)), p=self.dropout, training=True)
        h = F.dropout(F.relu(self.l2(h)), p=self.dropout, training=True)
        return self.l3(h)

# ────────────────────────────────────────────────
# Standard training with dropout
# ────────────────────────────────────────────────
N = 40
x = torch.linspace(-3, 3, N).unsqueeze(1)
y = torch.sin(x) + 0.1*torch.randn_like(x)

model = DropoutNN(hidden=100, dropout=0.1)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)
for epoch in range(3000):
    opt.zero_grad()
    pred = model(x)
    loss = F.mse_loss(pred, y)
    loss.backward(); opt.step()

# ────────────────────────────────────────────────
# MC Dropout at test time: KEEP dropout ON
# ────────────────────────────────────────────────
x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
T = 200

model.train()  # dropout stays on!
preds = []
with torch.no_grad():
    for _ in range(T):
        preds.append(model(x_test).numpy().flatten())
preds = np.array(preds)

mean = preds.mean(axis=0)
std = preds.std(axis=0)

# ────────────────────────────────────────────────
# Compare: deterministic (eval mode) vs MC
# ────────────────────────────────────────────────
model.eval()  # dropout off
with torch.no_grad():
    deterministic = model(x_test).numpy().flatten()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(x_test.numpy(), deterministic, 'b-', lw=2, label='Deterministic (no dropout)')
axes[0].scatter(x.numpy(), y.numpy(), s=15, color='red', label='data')
axes[0].plot(x_test.numpy(), np.sin(x_test.numpy()), 'k--', alpha=0.5)
axes[0].set_title('Standard NN (point prediction)')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].fill_between(x_test.numpy().flatten(), mean - 2*std, mean + 2*std, alpha=0.3, label='±2σ (MC Dropout)')
axes[1].plot(x_test.numpy(), mean, 'b-', lw=2, label='MC mean')
axes[1].scatter(x.numpy(), y.numpy(), s=15, color='red', label='data')
axes[1].plot(x_test.numpy(), np.sin(x_test.numpy()), 'k--', alpha=0.5)
axes[1].set_title(f'MC Dropout (T={T}) — uncertainty in OOD')
axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig('mc_dropout.png', dpi=150); plt.show()

# Aleatoric + epistemic
aleatoric = 0.1**2  # from training noise assumption
epistemic = std**2
print(f"Mean epistemic std (center): {std[80:120].mean():.4f}")
print(f"Mean epistemic std (OOD, x>4): {std[180:].mean():.4f}")
```

**관찰**: Training region 내부는 std 작음, OOD에서 std 폭증 — 원하는 Bayesian 거동.

---

## 🔗 AI/ML 연결

### Medical Imaging
Camelyon16 등 benchmarks: MC Dropout ResNet으로 pathology segmentation + uncertainty map.

### Autonomous Driving
Perception system의 out-of-distribution detection에 MC Dropout 활용.

### Reinforcement Learning
Exploration bonus = MC Dropout variance (Osband et al. 2016).

### Active Learning
BALD (Bayesian Active Learning by Disagreement) = mutual information via MC Dropout samples.

### Deep Ensembles
Lakshminarayanan et al. 2017 — multiple independent NNs. MC Dropout의 discrete 대안.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Dropout 있는 NN | No-dropout model에 적용 불가 |
| Bernoulli mask = good posterior approx | Limited expressiveness |
| Test-time dropout | Inference cost $T$배 증가 |
| Constant $p$ throughout | Layer-specific optimal $p$ 있음 (Concrete Dropout) |

**MC Dropout의 논쟁**:
- Osband (2016): MC Dropout의 epistemic uncertainty가 **data 많아져도 감소 안 함** — 진짜 epistemic 아님
- Gal et al. response: 실전 성능은 잘 나옴, approximation의 한 종류
- 실전: 여전히 **most popular BNN approach**, 대부분 응용에서 충분.

---

## 📌 핵심 정리

$$\boxed{\text{Dropout } p \iff q(W) = \prod(p\delta_0 + (1-p)\delta_M)}$$

$$\boxed{p(y^*|x^*, D) \approx \frac{1}{T}\sum_t p(y^*|x^*, W^{(t)})}$$

핵심:
- Standard dropout NN = BNN (Gal 2016)
- Test-time dropout 유지 + $T$ samples
- "Free BNN" — 추가 학습 불필요
- 구현 극도로 간단 (`model.train()` at inference)

---

## 🤔 생각해볼 문제

**문제 1** (기초): 표준 dropout과 MC Dropout의 구현 차이는?

<details>
<summary>해설</summary>

**Standard**: `model.eval()` at test, dropout off, single forward pass. Point prediction.

**MC Dropout**: `model.train()` (or explicit dropout flag) at test, dropout on, **$T$ forward passes**, mean + variance.

PyTorch:
```python
model.train()  # KEEP dropout on!
preds = [model(x) for _ in range(T)]
```

단 한 줄 차이로 BNN 됨. "Free" 이유.

Tensorflow/Keras: `training=True` argument to layer.

</details>

**문제 2** (심화): MC Dropout의 epistemic uncertainty가 "진짜 Bayesian이 아니다"라는 비판의 근거?

<details>
<summary>해설</summary>

Osband (2016) 비판:
- 진짜 epistemic은 **data 많아지면 → 0**
- MC Dropout variance는 $p$에 의존, **data 양에 둔감**
- Bernoulli posterior family가 **너무 제한적**

실험: large data로도 MC Dropout variance가 유의미한 값 유지 → "진짜 epistemic"의 수학 정의 위배.

**옹호**:
- 실전 calibration에선 작동 (empirical)
- Approximate VI이므로 Gal이 증명한 그대로임
- "Approximation의 의의는 exactness가 아닌 usefulness"

**결론**: MC Dropout은 **"실용적 heuristic with variational interpretation"** — 완벽 Bayesian은 아니지만 실전 유용. Critical application은 다른 방법 병용.

</details>

**문제 3** (AI 연결): GPT 같은 LLM에 dropout이 이미 있음. Inference 시 dropout 켜고 multiple generation → Bayesian?

<details>
<summary>해설</summary>

**이론**: Yes — LLM's dropout도 variational interpretation 가능.

**실전**:
- 대부분 LLM inference는 `model.eval()` (dropout off)
- Dropout on으로 multiple generation → **generation variance ≈ epistemic uncertainty**
- **Hallucination detection**에 유용: high variance → 답이 uncertain

**연구**:
- Kuhn et al. (2023) "Semantic Uncertainty": multiple generation의 semantic variance로 hallucination 감지
- Malinin & Gales (2021): ensemble-based uncertainty for structured prediction

**Caveat**: 
- LLM sampling은 이미 stochastic (temperature) — dropout 효과와 혼합
- Temperature vs dropout uncertainty를 분리하는 것이 open problem

**Practical**: GPT에 `temperature=0.7` + $T=10$ generation → disagreement가 uncertainty proxy. MC Dropout의 natural LLM extension.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Variational BNN — Bayes by Backprop](./03-variational-bnn.md) | [📚 README로 돌아가기](../README.md) | [05. SWAG와 SGD의 Bayesian 해석 ▶](./05-swag-sgld.md) |

</div>
