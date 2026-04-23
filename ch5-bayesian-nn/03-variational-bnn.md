# 03. Variational BNN — Bayes by Backprop

## 🎯 핵심 질문

- Factorized Gaussian $q_\phi(W) = \prod_i \mathcal{N}(w_i; \mu_i, \sigma_i^2)$로 posterior 근사?
- **Reparameterization + ELBO gradient**로 Bayesian NN의 end-to-end 학습?
- 각 weight의 $\sigma_i$가 어떻게 uncertainty를 **encode**하는가?
- Scale prior(Gaussian, Laplace, mixture) 선택의 영향?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

Blundell et al. (2015)의 "Bayes by Backprop"이 **BNN의 표준 VI 접근**. MC Dropout(Ch5-04)보다 더 명시적이고 flexible. Weight별 uncertainty로 **pruning, compression, active learning** 가능. 현대 Pyro, TensorFlow Probability의 BNN API가 이 알고리즘 기반.

---

## 📐 수학적 선행 조건

- [Ch2-01~05](../ch2-variational-inference/01-vi-idea-elbo.md): VI, reparam trick
- [Ch5-01 BNN](./01-bnn-formulation.md)
- ELBO의 분해 (2) (Ch2-02)

---

## 📖 직관적 이해

### 핵심 아이디어

Each weight $w_{ij}$는 **scalar에서 Gaussian** $\mathcal{N}(\mu_{ij}, \sigma_{ij}^2)$로 바꿈. 학습 파라미터가 **2배**.

ELBO:
$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(W)}[\log p(D|W)] - \text{KL}(q_\phi(W)\|p(W))$$

Reparam: $w_{ij} = \mu_{ij} + \sigma_{ij}\epsilon_{ij}, \epsilon\sim\mathcal{N}(0, 1)$.

### 훈련 과정

```
For each mini-batch:
   1. Sample ε ~ N(0, 1) for each weight
   2. Compute w = μ + σ ⊙ ε (reparam)
   3. Forward pass: y_pred = f_w(x)
   4. Loss = -log p(y | y_pred) + KL(q || prior)
   5. Backprop to (μ, σ)
```

### Uncertainty 해석

- **$\mu_{ij}$ 근처** $\sigma_{ij}$ 작음 → weight 거의 확정
- **큰 $\sigma_{ij}$** → weight 불확실 → pruning 후보
- Test 시: $W$ 샘플링으로 predictive uncertainty

### 요리 비유

"각 재료 양을 **범위**로 지정". 각 조리마다 범위 내 실제 값을 뽑아서 요리 → 일관성과 다양성의 균형.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Variational Posterior over Weights

$$q_\phi(W) = \prod_{ij} \mathcal{N}(w_{ij}; \mu_{ij}, \sigma_{ij}^2)$$

$\phi = \{\mu_{ij}, \sigma_{ij}\}_{ij}$. $\sigma_{ij} > 0$ 보장: $\sigma_{ij} = \log(1 + \exp(\rho_{ij}))$ (softplus).

### 정의 3.2 — ELBO for BNN

$$\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(W)}[\log p(D|W)] - \text{KL}(q_\phi(W)\|p(W))$$

첫 항: reparam + MC. 둘째 항: factorized → **per-weight KL의 합** (closed form if Gaussian).

### 정의 3.3 — KL Term (Gaussian Prior)

$p(W) = \prod_{ij}\mathcal{N}(0, \tau^2)$:

$$\text{KL}(q\|p) = \sum_{ij}\left[\log\frac{\tau}{\sigma_{ij}} + \frac{\sigma_{ij}^2 + \mu_{ij}^2}{2\tau^2} - \frac{1}{2}\right]$$

### 정의 3.4 — Scale Mixture of Gaussians Prior

Blundell et al.:
$$p(w) = \pi\mathcal{N}(0, \sigma_1^2) + (1-\pi)\mathcal{N}(0, \sigma_2^2)$$

$\sigma_1 \gg \sigma_2$: 두 scale 혼합. Sparsity 유도.

---

## 🔬 정리와 증명

### 정리 3.1 — Per-Weight KL Decomposition

**명제**: Factorized $q, p$ 하:

$$\text{KL}(q(W)\|p(W)) = \sum_{ij}\text{KL}(q(w_{ij})\|p(w_{ij}))$$

**증명**: 
$$\text{KL}(q\|p) = \int \prod q_{ij}(\log\prod q_{ij} - \log\prod p_{ij}) = \sum_{ij}\int q_{ij}\log\frac{q_{ij}}{p_{ij}}$$

$\square$

### 정리 3.2 — Bayes by Backprop Gradient

**명제**: Reparam $w = \mu + \sigma\epsilon$에서:

$$\nabla_\phi \mathcal{L} = \mathbb{E}_{\epsilon}\left[\nabla_\phi \left(\log p(D|W(\phi, \epsilon))\right)\right] - \nabla_\phi \text{KL}(q_\phi\|p)$$

둘 다 직접 계산 가능 (first term은 standard backprop, second는 closed form).

**증명**: Ch2-05 정리 5.1 + KL closed-form. $\square$

### 정리 3.3 — Mini-Batch Scaled ELBO

**명제**: Mini-batch $B \subset D$에서 unbiased ELBO estimate:

$$\tilde{\mathcal{L}}(\phi) = \frac{|D|}{|B|}\mathbb{E}_{q}\left[\sum_{i \in B}\log p(y_i|x_i, W)\right] - \text{KL}(q_\phi\|p)$$

$\mathbb{E}[\tilde{\mathcal{L}}] = \mathcal{L}$.

**KL scaling**: 일부 구현은 $\beta \cdot \text{KL}$ with $\beta = |B|/|D|$ ← 에서 시작해 1까지 증가 (KL annealing).

### 정리 3.4 — Local Reparameterization Trick (Kingma et al. 2015)

**명제**: 각 weight별 $\epsilon$ sampling 대신 **layer별 pre-activation에 직접 reparam**:

$$z_j = \sum_i w_{ij}x_i = \sum_i(\mu_{ij}+\sigma_{ij}\epsilon_{ij})x_i$$

$$= \mu_{z_j} + \sigma_{z_j}\xi_j$$

where $\mu_{z_j} = \sum_i\mu_{ij}x_i, \sigma_{z_j}^2 = \sum_i\sigma_{ij}^2 x_i^2, \xi_j \sim \mathcal{N}(0, 1)$.

**장점**: Batch 내 different samples가 different $\xi$ 사용 → gradient variance 감소.

**증명**: Linear combination of independent Gaussians = Gaussian. $\square$

### 정리 3.5 — Predictive Distribution

**명제**: 학습 후, $x^*$에 대한 predictive:

$$p(y^*|x^*, D) \approx \frac{1}{T}\sum_{t=1}^T p(y^*|x^*, W^{(t)}), \quad W^{(t)} \sim q_\phi^*(W)$$

$T \in [10, 100]$ 정도.

---

## 💻 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

class BayesianLinear(nn.Module):
    def __init__(self, in_f, out_f, prior_sigma=1.0):
        super().__init__()
        self.in_f, self.out_f = in_f, out_f
        # Parameters: mu and rho (σ = softplus(ρ))
        self.mu_w = nn.Parameter(torch.randn(out_f, in_f) * 0.1)
        self.rho_w = nn.Parameter(torch.full((out_f, in_f), -5.0))
        self.mu_b = nn.Parameter(torch.zeros(out_f))
        self.rho_b = nn.Parameter(torch.full((out_f,), -5.0))
        self.prior_sigma = prior_sigma

    def forward(self, x):
        sigma_w = F.softplus(self.rho_w)
        sigma_b = F.softplus(self.rho_b)
        eps_w = torch.randn_like(sigma_w)
        eps_b = torch.randn_like(sigma_b)
        w = self.mu_w + sigma_w * eps_w
        b = self.mu_b + sigma_b * eps_b
        return F.linear(x, w, b)

    def kl(self):
        # KL(N(μ,σ²) || N(0, prior_sigma²))
        sigma_w = F.softplus(self.rho_w); sigma_b = F.softplus(self.rho_b)
        tau = self.prior_sigma
        kl_w = (torch.log(tau/sigma_w) + (sigma_w**2 + self.mu_w**2)/(2*tau**2) - 0.5).sum()
        kl_b = (torch.log(tau/sigma_b) + (sigma_b**2 + self.mu_b**2)/(2*tau**2) - 0.5).sum()
        return kl_w + kl_b

class BayesianMLP(nn.Module):
    def __init__(self, hidden=50):
        super().__init__()
        self.l1 = BayesianLinear(1, hidden)
        self.l2 = BayesianLinear(hidden, 1)
    def forward(self, x):
        h = torch.tanh(self.l1(x))
        return self.l2(h)
    def kl(self):
        return self.l1.kl() + self.l2.kl()

# Data
N = 50
x = torch.linspace(-3, 3, N).unsqueeze(1)
y = torch.sin(x) + 0.1*torch.randn_like(x)

model = BayesianMLP(hidden=50)
opt = torch.optim.Adam(model.parameters(), lr=1e-2)
sigma_noise = 0.1

n_epochs = 3000
kl_weight = 1.0 / N          # per-example balance
for epoch in range(n_epochs):
    opt.zero_grad()
    pred = model(x)
    log_lik = -0.5*((y - pred)**2 / sigma_noise**2).sum()
    kl = model.kl()
    elbo = log_lik - kl_weight * kl
    loss = -elbo
    loss.backward(); opt.step()
    if epoch % 500 == 0:
        print(f"ep {epoch}: ELBO={elbo.item():.2f}, KL={kl.item():.2f}")

# Predict with MC samples
x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
preds = []
with torch.no_grad():
    for _ in range(200):
        preds.append(model(x_test).numpy().flatten())
preds = np.array(preds)
mean = preds.mean(axis=0)
lo, hi = np.percentile(preds, [2.5, 97.5], axis=0)

plt.figure(figsize=(10, 5))
plt.fill_between(x_test.numpy().flatten(), lo, hi, alpha=0.3, label='95% CI')
plt.plot(x_test.numpy(), mean, 'b-', lw=2, label='Posterior mean')
plt.scatter(x.numpy(), y.numpy(), s=15, color='red', label='data')
plt.plot(x_test.numpy(), np.sin(x_test.numpy()), 'k--', alpha=0.5, label='true sin')
plt.legend(); plt.grid(alpha=0.3)
plt.title('Bayes by Backprop — uncertainty expands OOD')
plt.savefig('bbb.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Deep Reinforcement Learning
Bayesian Q-networks (Osband 2016), Thompson sampling in deep RL.

### Continual Learning
Variational continual learning (Nguyen et al. 2018) — 이전 task's posterior가 prior.

### Active Learning
$\sigma_{ij}$로 각 weight uncertainty 정량화 → acquisition function.

### Pyro/TensorFlow Probability
`pyro.distributions.Normal`로 weight 분포 정의 → 자동 ELBO.

### Natural Gradient VI (Ch2-04)
Bayes by Backprop의 **improved optimizer** — Khan et al. 2018.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Factorized Gaussian | Weight correlation 무시 |
| Prior specification | Scale 민감 |
| Mini-batch ELBO unbiased | Gradient variance 여전 |
| Single mode | Multimodal posterior 못 잡음 |

**실무 팁**:
- **$\rho$ 초기화**: $\rho_0 = -5$ or $-3$ (small $\sigma$부터 시작)
- **KL annealing**: 초기엔 KL 작게 → 점차 증가 (collapse 방지)
- **MC samples per batch**: 1이면 fast, 10이면 stable
- **Local reparam** (정리 3.4): layer-wise variance 감소

---

## 📌 핵심 정리

$$\boxed{q_\phi(W) = \prod_{ij}\mathcal{N}(\mu_{ij}, \sigma_{ij}^2)}$$

$$\boxed{\mathcal{L} = \mathbb{E}_q[\log p(D|W)] - \text{KL}(q\|p)}$$

핵심:
- **2× parameters** (mean + log-σ)
- End-to-end backprop + reparam
- Per-weight KL closed form (Gaussian-Gaussian)
- **Local reparameterization**으로 gradient variance 감소

---

## 🤔 생각해볼 문제

**문제 1** (기초): Trained BBB에서 $\sigma_{ij}$가 매우 작은 weight는 무엇을 의미?

<details>
<summary>해설</summary>

**$\sigma_{ij} \to 0$**: weight가 $\mu_{ij}$로 거의 확정. Posterior가 delta처럼 집중. 두 해석:

1. **Data가 weight 값을 매우 확실히 제약** → 정보 많음
2. **Prior가 $\mu_{ij}$ 주변을 강하게 preferring**

반면 **$\sigma_{ij}$ 큼** = prior와 비슷 = **unused weight** → pruning 후보.

**Practical**: $\sigma_{ij} / \mu_{ij}$ 비율로 **weight importance** 측정. 낮은 비율 → remove.

이것이 **Bayesian pruning** (Louizos et al. 2017)의 기초.

</details>

**문제 2** (심화): BBB의 gradient variance를 줄이는 기법들?

<details>
<summary>해설</summary>

1. **Local reparameterization** (정리 3.4): per-layer pre-activation reparam
2. **Multiple MC samples per batch**: $T > 1$
3. **Flipout** (Wen et al. 2018): pseudo-independent noise per example within batch
4. **Antithetic sampling**: $\epsilon$과 $-\epsilon$ 쌍 사용
5. **Control variate** (Ch2-06): baseline subtraction

최근 **Implicit variational methods**(SVGD, particle-based)는 이런 문제를 다른 angle로 해결.

</details>

**문제 3** (AI 연결): LLM fine-tuning에 Bayes by Backprop을 적용하려면?

<details>
<summary>해설</summary>

**Challenge**: 2× parameters, memory 과다.

**LoRA Bayes by Backprop**:
- LoRA adapter ($10^6$ params) 에만 Bayesian
- Mean/std = $2 \times 10^6$ (base LLM은 고정)
- Computationally feasible

**응용**:
- RLHF with uncertainty-aware reward
- Robust text generation with confidence
- Multi-task adapter with Bayesian posterior

**현재 frontier**: **Bayesian LoRA** + KL regularization on adapter → fine-tuned LLM의 reliable calibration.

**Drawback**: BBB는 mean-field → LLM의 복잡 correlation 놓칠 수 있음. Flow-based or KFAC 대체 탐색 중.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Laplace Approximation](./02-laplace-approximation.md) | [📚 README로 돌아가기](../README.md) | [04. MC Dropout = Approximate VI ▶](./04-mc-dropout.md) |

</div>
