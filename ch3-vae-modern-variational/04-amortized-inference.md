# 04. Amortized Inference

## 🎯 핵심 질문

- **Amortized inference**와 **non-amortized (per-datum) VI**는 어떻게 다른가?
- **Amortization gap** $\text{ELBO}^* - \text{ELBO}_{q_\phi}$는 왜 생기는가?
- 계산 trade-off — 왜 amortization이 대규모에서 필수인가?
- **Semi-amortized** VI (Kim et al. 2018)는 어떻게 둘의 장점을 결합하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

VAE(Ch3-01)의 encoder가 바로 amortized inference. 각 $x_i$마다 새로운 VI를 돌리지 않고 **하나의 NN이 모든 $x$**에 대해 $q(z|x)$를 produce. 이것이 **inference 비용을 상수 시간**으로 만들고, 실시간 deployment 가능. Image generation, representation learning의 실전적 조건. 그러나 **모든 $x$에 한 번에 최적화**의 제약이 정확도를 낮춤 (amortization gap).

---

## 📐 수학적 선행 조건

- [Ch3-01 VAE](./01-vae-derivation.md)
- [Ch2-01~03](../ch2-variational-inference/01-vi-idea-elbo.md): VI, CAVI
- Function approximation theory basics

---

## 📖 직관적 이해

### Non-amortized vs Amortized

| | Non-amortized (per-datum) | Amortized |
|---|---|---|
| $q$ 형태 | 각 $x_i$마다 $q_i$ 별도 | 하나의 NN $q_\phi(z\|x)$ |
| 파라미터 수 | $N$-scale (데이터 의존) | **상수** (NN 크기) |
| 학습 비용 | 매 데이터마다 VI iteration | NN 한 번 forward |
| Inference (new $x$) | VI from scratch | **Constant time** forward |
| 정확도 | Tight (local 최적화) | **Gap**이 있음 (NN의 limited capacity) |

### Amortization gap

각 $x$마다 non-amortized $q^*_x$가 optimal이라 하자:
$$q^*_x = \arg\max_{q\in\mathcal{Q}} \text{ELBO}(x; q)$$

Amortized $q_\phi(z|x)$는 **NN parameter $\phi$ 공유** — 제약.

$$\text{Amortization gap}(x) = \text{ELBO}(x; q^*_x) - \text{ELBO}(x; q_\phi(\cdot|x))$$

항상 $\geq 0$.

### 요리 비유

- Non-amortized: "손님마다 요리사를 새로 고용해서 최적 메뉴 연구"
- Amortized: "메뉴 결정 NN 한 명을 훈련" — 모든 손님 빠르게 응대하지만 개인 맞춤 완벽하진 못함

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Non-amortized VI

각 $x_i$에 대해:
$$q^*_i = \arg\max_{q \in \mathcal{Q}_i}\mathcal{L}(x_i; q)$$

$\mathcal{Q}_i$는 per-example family (예: Gaussian with $\mu_i, \sigma_i$).

### 정의 4.2 — Amortized VI

Inference network $q_\phi(z|x)$ (NN with param $\phi$):
$$\phi^* = \arg\max_\phi \frac{1}{N}\sum_i \mathcal{L}(x_i; q_\phi(\cdot|x_i))$$

### 정의 4.3 — Amortization Gap

$$g(x; \phi) = \mathcal{L}(x; q^*_x) - \mathcal{L}(x; q_\phi(\cdot|x)) \geq 0$$

Average amortization gap:
$$G(\phi) = \mathbb{E}_{x}[g(x; \phi)]$$

### 정의 4.4 — Approximation Gap

$$a(x) = \log p(x) - \mathcal{L}(x; q^*_x) = \text{KL}(q^*_x\|p(\cdot|x)) \geq 0$$

Variational family의 본질적 제약 (Ch2-01 정리 1.5).

### 정의 4.5 — Total Inference Gap

$$\log p(x) - \mathcal{L}(x; q_\phi(\cdot|x)) = \underbrace{a(x)}_{\text{approximation}} + \underbrace{g(x; \phi)}_{\text{amortization}}$$

---

## 🔬 정리와 증명

### 정리 4.1 — 총 gap 분해

**명제**: Amortized VI의 전체 inference error는:

$$\log p(x) - \mathcal{L}_{\text{amortized}}(x) = a(x) + g(x; \phi)$$

- $a(x)$: family 제약으로 인한 gap
- $g(x; \phi)$: amortization으로 인한 gap

**증명**: 정의 4.3, 4.4에서 직접. $\square$

**귀결**: Inference quality 개선 방법:
1. **Family 확장** ($\mathcal{Q}$ → normalizing flow, full-rank Gaussian) → $a$ 감소
2. **Inference network 표현력** ($q_\phi$ NN deeper/wider) → $g$ 감소
3. **둘 다** (Flow + amortization)

### 정리 4.2 — Amortization Gap은 항상 ≥ 0

**명제**: $\forall x, \phi: g(x; \phi) \geq 0$.

**증명**: $q_\phi(\cdot|x) \in \mathcal{Q}$이므로 $\mathcal{L}(x; q_\phi) \leq \max_q \mathcal{L}(x; q) = \mathcal{L}(x; q^*_x)$. $\square$

### 정리 4.3 — Semi-Amortized VI (Kim et al. 2018)

**아이디어**: Amortized $q_\phi$를 initial point로 쓰고, 각 $x$에 대해 **몇 steps의 local VI** refinement:

```
q_0 ← q_φ(·|x)           (amortized init)
For t = 1..T:
   q_t ← gradient step on ELBO(x; q_t-1)
Use q_T for this x
```

**정리**: 이 절차가 amortization gap을 **$T$ step에 걸쳐** 감소시킴.

**증명**: Gradient ascent ensures $\mathcal{L}(x; q_{t+1}) \geq \mathcal{L}(x; q_t)$ (with sufficiently small step). Limit as $T \to \infty$ recovers $q^*_x$. $\square$

**Trade-off**: Inference 비용이 $T$배 증가 ($T$는 보통 1~5).

### 정리 4.4 — Encoder Capacity와 Gap

**명제**: Inference network $q_\phi$의 NN이 deeper·wider일수록 amortization gap이 감소 (empirically 보임; Cremer et al. 2018).

**heuristic**: Universal function approximator로서 NN이 어떤 $x \mapsto q^*_x$ 함수도 **점근적으로 근사 가능**. 실전에서 capacity가 제한되면 gap 남음.

---

## 💻 PyTorch 구현 — Semi-amortized VAE

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SemiAmortizedVAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400, latent=20, n_svi=3, svi_lr=0.01):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, input_dim)
        self.n_svi = n_svi
        self.svi_lr = svi_lr

    def encode(self, x):
        h = F.relu(self.enc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z):
        h = F.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(h))

    def elbo(self, x, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        recon = self.decode(z)
        BCE = F.binary_cross_entropy(recon, x, reduction='none').sum(dim=1)
        KLD = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1)
        return -(BCE + KLD)  # ELBO per example

    def forward(self, x):
        mu, logvar = self.encode(x)
        # Semi-amortized refinement: few SVI steps starting from (mu, logvar)
        mu = mu.detach().clone().requires_grad_(True)
        logvar = logvar.detach().clone().requires_grad_(True)
        for _ in range(self.n_svi):
            elbo = self.elbo(x, mu, logvar).sum()
            grads = torch.autograd.grad(elbo, [mu, logvar], create_graph=True)
            mu = mu + self.svi_lr * grads[0]
            logvar = logvar + self.svi_lr * grads[1]
        final_elbo = self.elbo(x, mu, logvar)
        return final_elbo, mu, logvar

# Comparison script (pseudo):
# amortized VAE: n_svi=0
# semi-amortized: n_svi=3
# non-amortized (standalone): n_svi=1000
# → report ELBO on held-out test set
```

**일반적 결과**:
- Amortized (n=0): ELBO = $a$
- Semi-amortized (n=3): ELBO gain ~5–10%
- Non-amortized (n=1000): 상한, 하지만 **1000× 느림**

---

## 🔗 AI/ML 연결

### VAE Training
표준 VAE는 순수 amortized. 대규모 이미지/텍스트에서 필수.

### Iterative VAE / Semi-amortized VAE
Encoder의 "초기화"로 사용, 몇 SVI step → 연구 분야.

### Meta-Learning 관점
Amortization = "모든 task의 공통 inference 학습" — Meta-learning과 밀접. 각 $x$가 new task.

### Diffusion Model
Diffusion은 "per-step amortized inference" — 각 $t$에 대해 공통 network.

### Retrieval-augmented Generation
RAG는 inference에 외부 검색 → amortized encoder를 **data-dependent**로 확장.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Encoder NN 표현력 | Limited capacity → amortization gap |
| Training data가 test distribution 대표 | OOD에서 gap 커짐 |
| Constant-time inference | Encoder가 gigantic해지면 non-negligible |
| Gap이 fine으로 수렴 | 복잡 posterior에선 깊은 network 필요 |

**실무 팁**:
- **Gap 진단**: training set에서 amortized ELBO vs test-time optimized ELBO 비교
- **Large gap** = encoder 용량 부족 → deeper NN 또는 flow-based posterior

---

## 📌 핵심 정리

$$\boxed{\log p(x) - \mathcal{L}_{\text{amortized}}(x) = \underbrace{a(x)}_{\text{approximation}} + \underbrace{g(x; \phi)}_{\text{amortization}}}$$

핵심:
- Amortization = **inference 공유** by NN
- Gap은 encoder 용량·family 표현력에 의존
- Semi-amortized = amortized init + few local steps
- 대규모 deploy엔 **amortization이 사실상 필수**

---

## 🤔 생각해볼 문제

**문제 1** (기초): Amortized VI에서 $\phi$의 parameter 수가 $N$(데이터 수)과 **무관**한 것이 왜 중요한가?

<details>
<summary>해설</summary>

Non-amortized: 각 $x_i$마다 $(\mu_i, \sigma_i)$ → $2d \cdot N$ 파라미터. $N = 10^6$, $d = 100$이면 $2\times 10^8$.

Amortized: encoder NN 파라미터만 (예: $10^6$) — $N$에 무관.

이것이 **scalability**의 열쇠. Mini-batch SGD가 $N$-independent complexity per step.

또한 **new $x$ (test time)** 에도 즉시 적용 가능 — non-amortized는 **매번 VI 다시**.

</details>

**문제 2** (심화): Amortization gap이 **클수록** VAE 성능에 어떤 영향?

<details>
<summary>해설</summary>

- **ELBO가 낮음** → $\log p(x)$의 loose bound
- Training loss가 **왜곡** — amortized ELBO 최적화가 true likelihood와 괴리
- **Generation**: decoder $p_\theta$가 정확한 posterior 반영 못 함 → blurry/distorted samples
- **Representation**: $q_\phi(z|x)$가 ideal posterior 대비 bias된 latent → downstream task 영향

Cremer et al. (2018)의 연구: amortization gap이 VAE 생성 품질의 **주요 제약** 중 하나. Flow-based posterior나 semi-amortized가 실전에서 도움.

</details>

**문제 3** (AI 연결): Meta-learning의 **MAML** (Model-Agnostic Meta-Learning)과 amortized VI는 철학적으로 어떻게 닮았는가?

<details>
<summary>해설</summary>

| | MAML | Amortized VI |
|---|---|---|
| 공유 | Initial $\theta_0$ | Encoder $\phi$ |
| Task/sample | Task-specific fine-tune | Per-$x$ inference |
| Amortization | Few-step adaptation에 친화적 | Encoder가 inference 공유 |
| Objective | $\min_\theta \mathbb{E}_{\text{task}}[\mathcal{L}(\theta - \alpha\nabla\mathcal{L})]$ | $\min_\phi \mathbb{E}_x[-\text{ELBO}(x; q_\phi)]$ |

**Semi-amortized VI** = "MAML applied to VI" (Kim et al. 2018 implicit): amortized init + few gradient steps = few-shot adaptation.

두 접근 모두 "**shared basis + local refinement**" 철학. Meta-learning literature와 variational inference literature가 이 지점에서 convergence.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Normalizing Flows](./03-normalizing-flows.md) | [📚 README로 돌아가기](../README.md) | [05. Importance-weighted VAE (IWAE) ▶](./05-iwae.md) |

</div>
