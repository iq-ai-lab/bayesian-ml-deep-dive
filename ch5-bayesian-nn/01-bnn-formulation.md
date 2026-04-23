# 01. BNN의 수학적 정식화

## 🎯 핵심 질문

- NN 가중치 $W$에 대한 **prior** $p(W)$와 **likelihood** $p(D|W)$는 구체적으로 무엇인가?
- Posterior $p(W|D) = p(D|W)p(W)/p(D)$가 왜 **고차원 BNN에서 intractable**한가?
- 예측 $p(y^*|x^*, D) = \int p(y^*|x^*, W)p(W|D)dW$를 어떻게 근사하는가?
- 왜 일반 MCMC가 수백만 파라미터 BNN에 실패하는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**Point-estimate NN**(일반 학습된 NN)은 confidence 못 주고 OOD에서 overconfident. BNN은 이 문제의 **수학적으로 정확한 해결** — 불확실성을 모델에 내장. 자율주행·의료진단 같은 안전-critical 응용에서 필수. BNN의 정식화가 Laplace(Ch5-02), Bayes by Backprop(Ch5-03), MC Dropout(Ch5-04), SWAG/SGLD(Ch5-05)의 공통 출발점.

---

## 📐 수학적 선행 조건

- [Ch1 전체](../ch1-bayesian-foundation/01-bayes-rule-four-roles.md): Bayesian 기초, predictive
- [Ch2, Ch3](../ch2-variational-inference/01-vi-idea-elbo.md): VI, ELBO
- [Ch4](../ch4-mcmc/01-metropolis-hastings.md): MCMC
- NN basics (forward pass, backprop)

---

## 📖 직관적 이해

### Point NN vs BNN

**Point NN**: $W$는 **단일 값** (SGD로 얻음). 예측 $p(y^*|x^*, W)$.

**BNN**: $W$는 **분포** $p(W|D)$. 예측 $\int p(y^*|x^*, W)p(W|D)dW$ → marginalization over weight uncertainty.

### 확률모형 구조

```
Prior:       W ~ p(W) = N(0, σ²I)   (예: Gaussian)
Likelihood:  y|x, W ~ p(y|x, W) = N(f_W(x), σ_n²)  (회귀)
                               or Categorical(softmax(f_W(x)))  (분류)
Posterior:   W|D ∝ p(D|W) p(W)
Predictive:  y*|x*, D = ∫ p(y*|x*, W) p(W|D) dW
```

### 왜 Intractable?

- $W \in \mathbb{R}^d$, $d \sim 10^6$ ~ $10^9$
- Posterior는 **$d$차원 분포** — 일반 MCMC로 다루기 어려움
- Evidence $p(D) = \int p(D|W)p(W)dW$ = $d$차원 적분 → 절대 직접 계산 불가
- Loss landscape는 **다봉, 비볼록**

### 요리 비유

- Point NN: "**한 명의 요리사**가 주방 관리"
- BNN: "**요리사의 분포** — 각 요리사 확률만큼 기여"
- 예측: 각 요리사의 의견을 posterior 가중 평균

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Bayesian Neural Network

NN $f_W: \mathcal{X} \to \mathcal{Y}$ with weights $W \in \mathbb{R}^d$. **BNN**은:

1. **Prior** $p(W)$: 가장 흔한 $\mathcal{N}(0, \sigma^2 I)$
2. **Likelihood** $p(y|x, W)$:
   - 회귀: $\mathcal{N}(f_W(x), \sigma_n^2)$ or heteroscedastic $\mathcal{N}(\mu_W(x), \sigma_W^2(x))$
   - 분류: $\text{Categorical}(\text{softmax}(f_W(x)))$
3. **Posterior**: $p(W|D) \propto p(D|W)p(W) = \left[\prod_i p(y_i|x_i, W)\right] p(W)$
4. **Predictive**: $p(y^*|x^*, D) = \int p(y^*|x^*, W) p(W|D) dW$

### 정의 1.2 — Model Evidence (Marginal Likelihood)

$$p(D) = \int p(D|W) p(W) dW$$

Model comparison의 지표 (Ch1-01 정리 1.4).

### 정의 1.3 — Epistemic vs Aleatoric (Ch7-03 preview)

$$\text{Var}[y^*|x^*, D] = \underbrace{\mathbb{E}_{W|D}[\text{Var}(y^*|x^*, W)]}_{\text{aleatoric}} + \underbrace{\text{Var}_{W|D}[\mathbb{E}(y^*|x^*, W)]}_{\text{epistemic}}$$

BNN의 **핵심 결과물**.

---

## 🔬 정리와 증명

### 정리 1.1 — BNN Predictive의 MC 근사

**명제**:
$$p(y^*|x^*, D) \approx \frac{1}{T}\sum_{t=1}^T p(y^*|x^*, W^{(t)}), \quad W^{(t)} \sim p(W|D)$$

**증명**: MC integration. Law of large numbers → 수렴 (a.s.). $\square$

**실전**: $W^{(t)}$ 샘플 방법에 따라 — MC Dropout $T=50\sim100$, Ensemble $T=5\sim20$, HMC $T=\infty$ (full chain).

### 정리 1.2 — L2 Regularized NN = MAP BNN

**명제**: $p(W) = \mathcal{N}(0, \tau^2 I)$ prior 하에서:

$$\hat W_{MAP} = \arg\max_W [\log p(D|W) + \log p(W)] = \arg\min_W \left[-\log p(D|W) + \frac{1}{2\tau^2}\|W\|^2\right]$$

이것이 **L2-regularized NN training**. 즉 **일반 학습된 NN = MAP BNN**.

**증명**: Ch1-02 정리 2.2 적용. $\square$

**귀결**: "Bayesian이 아닌 SGD-trained NN"도 사실 BNN의 **한 극단** (delta posterior = MAP).

### 정리 1.3 — BNN의 Overparameterization과 Posterior 형태

**명제** (informal): $d \gg N$ (parameters ≫ data)이면:
- Likelihood가 "**high-dim ridge**"처럼 작동
- Posterior는 넓은 basin of solution을 가짐
- Prior가 **중요** (identifiability 회복)

**경험적 관찰**: BNN posterior는 **near-degenerate** — 많은 weight 조합이 같은 함수 산출. Degeneracy를 다루는 것이 BNN 추론의 난제.

### 정리 1.4 — BNN 추론의 계층적 구조

**계층**:
1. **Exact posterior** $p(W|D)$ — intractable
2. **Approximate posterior**:
   - VI: $q_\phi(W)$ (Ch5-03)
   - Laplace: Gaussian at MAP (Ch5-02)
   - MC Dropout: Bernoulli variational (Ch5-04)
   - MCMC samples (HMC, SGLD — 제한적)
3. **Ensemble**: multiple MAP from diff init (heuristic Bayesian)

**원리적 우열**: Exact > HMC > VI (various) > point estimate.
**실전 scalability**: 역순.

### 예시 — 1D Regression BNN

Data: $y_i = \sin(x_i) + 0.1\epsilon_i, x_i \in [0, 2\pi]$, $N = 20$.
Model: 2-layer MLP with $\sim$100 params.

- Point NN: 곡선 fit, 분산은 training noise만 반영
- BNN (Laplace): 곡선 + data-sparse region에서 uncertainty band 확대
- 새 $x^* = 4\pi$ (extrapolation): BNN이 **높은 uncertainty** 표시 → in/out-of-distribution 신호

---

## 💻 PyTorch 구현 — BNN Regression (Small-Scale MCMC)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ────────────────────────────────────────────────
# Toy 1D data
# ────────────────────────────────────────────────
N = 20
x_train = torch.linspace(-3, 3, N).unsqueeze(1)
y_train = torch.sin(x_train) + 0.1 * torch.randn(N, 1)

# ────────────────────────────────────────────────
# Small MLP (so HMC feasible)
# ────────────────────────────────────────────────
class SmallMLP(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, 1)
    def forward(self, x):
        h = torch.tanh(self.fc1(x))
        return self.fc2(h)

# Flatten/unflatten weights
def get_flat_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])
def set_flat_params(model, flat):
    idx = 0
    for p in model.parameters():
        n = p.numel()
        p.data = flat[idx:idx+n].view_as(p)
        idx += n

model = SmallMLP(hidden=16)
n_params = sum(p.numel() for p in model.parameters())
print(f"# params: {n_params}")

# ────────────────────────────────────────────────
# Log-posterior: log p(W|D) ∝ log p(D|W) + log p(W)
# Prior: N(0, 1), Likelihood: N(f(x), 0.1²)
# ────────────────────────────────────────────────
def log_posterior(flat_W):
    set_flat_params(model, flat_W)
    pred = model(x_train)
    log_lik = -0.5 * ((y_train - pred)**2 / 0.01).sum()
    log_prior = -0.5 * (flat_W**2).sum()     # prior N(0, 1)
    return log_lik + log_prior

# ────────────────────────────────────────────────
# HMC (Ch4-03) for small BNN
# ────────────────────────────────────────────────
def hmc_step(W, eps, L):
    W = W.clone().requires_grad_(True)
    p = torch.randn_like(W)
    log_p0 = log_posterior(W); log_p0.backward()
    H0 = -log_p0.item() + 0.5*(p**2).sum().item()
    W_new, p_new = W.detach().clone(), p.clone()
    for l in range(L):
        p_new = p_new + 0.5*eps*torch.autograd.grad(log_posterior(W_new), W_new.requires_grad_(True))[0]
        W_new = W_new.detach() + eps*p_new
        p_new = p_new + 0.5*eps*torch.autograd.grad(log_posterior(W_new.requires_grad_(True)), W_new)[0]
    H_new = -log_posterior(W_new.detach()).item() + 0.5*(p_new**2).sum().item()
    if np.log(np.random.rand()) < H0 - H_new:
        return W_new.detach(), True
    return W, False

# Warm start from a MAP-ish init + collect chain
W = torch.randn(n_params) * 0.1
samples = []
for it in range(2000):
    W, _ = hmc_step(W, eps=0.01, L=20)
    if it > 500:   # burn-in
        samples.append(W.clone())

# ────────────────────────────────────────────────
# Posterior predictive on test range
# ────────────────────────────────────────────────
x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
preds = []
for W in samples[-500:]:   # subsample
    set_flat_params(model, W)
    preds.append(model(x_test).detach().numpy().flatten())
preds = np.array(preds)

mean = preds.mean(axis=0)
lo, hi = np.percentile(preds, [2.5, 97.5], axis=0)

fig, ax = plt.subplots(figsize=(10, 5))
ax.fill_between(x_test.numpy().flatten(), lo, hi, alpha=0.3, label='95% predictive')
ax.plot(x_test.numpy(), mean, 'b-', lw=2, label='Posterior mean')
ax.scatter(x_train.numpy(), y_train.numpy(), s=30, color='red', label='data')
ax.plot(x_test.numpy(), np.sin(x_test.numpy()), 'k--', alpha=0.5, label='true sin')
ax.legend(); ax.grid(alpha=0.3)
ax.set_title('BNN with HMC — uncertainty expands in extrapolation')
plt.tight_layout(); plt.savefig('bnn_hmc.png', dpi=150); plt.show()
```

**관찰**: training 영역 내부는 tight band, 외부(extrapolation)는 uncertainty 확대 — 원하는 Bayesian 행동.

---

## 🔗 AI/ML 연결

### Laplace (Ch5-02)
Posterior 최빈값 주변 Gaussian 근사.

### Bayes by Backprop (Ch5-03)
Full VI로 posterior 근사.

### MC Dropout (Ch5-04)
Dropout NN을 variational BNN으로 해석.

### SWAG (Ch5-05)
SGD 궤적을 posterior Gaussian 근사로.

### Deep Ensembles
여러 NN 독립 학습 → 이산 BMA — "poor man's Bayesian".

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Prior $\mathcal{N}(0, I)$ | Scale 선택 민감 (Gaussian prior hyperparameter) |
| Likelihood 잘 specified | Model misspecification — posterior biased |
| Posterior가 유의미 | Over-parameterized → degenerate |
| MCMC 수렴 | $d = 10^6$에선 실패 → approximation 필수 |

**실전 체크리스트**:
- Prior variance: $\tau^2 \in [0.1, 10]$ 범위 스윕
- Likelihood noise $\sigma_n$: 학습 가능하게 or cross-val
- **Ensemble 보완**: 여러 독립 학습으로 multimodal 완화

---

## 📌 핵심 정리

$$\boxed{p(W|D) \propto p(D|W)p(W), \quad p(y^*|x^*, D) = \int p(y^*|x^*, W)p(W|D)dW}$$

핵심:
- Weights를 **확률변수**로 취급
- Evidence $p(D)$는 일반 BNN에서 **intractable**
- 추론: Laplace / VI / MC Dropout / SWAG / SGLD / HMC(작은 경우)
- 예측: posterior sample의 **MC average**
- Uncertainty = epistemic + aleatoric (Ch7-03)

---

## 🤔 생각해볼 문제

**문제 1** (기초): Prior $p(W) = \mathcal{N}(0, \tau^2 I)$에서 $\tau \to 0$과 $\tau \to \infty$의 극한은?

<details>
<summary>해설</summary>

**$\tau \to 0$**: prior가 0으로 집중 → posterior ≈ 0 → 모델이 데이터 무시, constant output. 강한 regularization = underfit.

**$\tau \to \infty$**: prior가 uniform(improper) → MAP = MLE → overfitting 위험.

**$\tau$ 선택**:
- Cross-validation
- Empirical Bayes: $\tau$를 data로 학습 (evidence 최대화)
- Hierarchical: $\tau$에 hyperprior, 함께 추론

실전: $\tau \in [0.1, 1]$이 흔 (with properly initialized NN).

</details>

**문제 2** (심화): BNN에서 **posterior degeneracy**(여러 weight가 같은 함수)가 inference에 미치는 영향은?

<details>
<summary>해설</summary>

**Degenerate posterior**: $W_1 \neq W_2$인데 $f_{W_1} = f_{W_2}$ (예: neuron permutation, scale-invariance).

**영향**:
- Weight-space MCMC: 사실상 **function-space MCMC**로 귀결 — 하지만 redundant weight dim 때문에 느림
- Laplace: Hessian이 singular → pseudo-inverse
- VI: mean-field는 degeneracy를 못 잡음
- 점추정(MAP)은 한 solution만 → 어떤 $W^*$든 같은 $f$이면 괜찮음

**실전 완화**:
- **Weight symmetry 사전 제거** (canonical form)
- **Function-space prior** (Ch5 advanced — GP-inspired)
- **Symmetry-invariant metrics** 사용

이것이 BNN posterior inference가 hierarchical/non-NN Bayesian보다 어려운 근본 이유.

</details>

**문제 3** (AI 연결): GPT-3의 파라미터 $1.75 \times 10^{11}$개에 대해 "Bayesian GPT"를 만들려면?

<details>
<summary>해설</summary>

**직접 HMC**: 불가능. $d = 10^{11}$에선 어떤 MCMC도 작동 안 함.

**현실적 접근**:
1. **Last-layer Bayesian**: 마지막 layer만 BNN (수백~수천 params)
2. **LoRA Bayesian**: Low-rank adapter에만 BNN (백만 params)
3. **SWAG over fine-tuning**: Fine-tuning trajectory로 Gaussian posterior
4. **MC Dropout**: 이미 dropout이 있으면 test-time에 유지 → free BNN
5. **Ensemble of fine-tuned**: discrete BMA

**응용**:
- Hallucination detection (epistemic uncertainty)
- Calibrated confidence (Ch7-04)
- Active learning (acquisition)

학술적 frontier: "**Fully Bayesian LLM**"은 미해결. BNN 이해가 이 분야 연구의 출발점.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch4-06 MCMC vs VI](../ch4-mcmc/06-mcmc-vs-vi.md) | [📚 README로 돌아가기](../README.md) | [02. Laplace Approximation ▶](./02-laplace-approximation.md) |

</div>
