# 01. GP 기반 BO 프레임워크

## 🎯 핵심 질문

- Black-box 함수 $f(x)$에 대한 **GP prior** $f \sim \mathcal{GP}(m, k)$는 어떻게 설정되는가?
- 관측 $\{(x_i, y_i)\}$ 후 **posterior predictive** $\mathcal{N}(\mu_n(x), \sigma_n^2(x))$가 Gaussian conditioning으로 유도?
- Acquisition function $a(x)$로 **다음 평가점**을 어떻게 선택?
- 전체 **BO 루프** (GP 업데이트 ↔ acquisition 최대화)의 수학적 구조?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**Hyperparameter tuning** (딥러닝 LR, architecture), **분자 설계**, **A/B testing 배치**, **로봇 제어**, **실험 설계** — 모두 "**비싼 black-box 함수 최적화**" 문제. BO는 GP의 posterior uncertainty로 **exploration-exploitation** 균형 → $O(\log T)$ calls로 near-optimal. Bayesian의 대표적 practical win.

---

## 📐 수학적 선행 조건

- [Ch1-03 Conjugate](../ch1-bayesian-foundation/03-conjugate-priors.md): Normal-Normal
- [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive): GP, RKHS
- Linear algebra: Cholesky decomposition
- Multivariate Gaussian conditioning

---

## 📖 직관적 이해

### GP의 아이디어

"함수에 대한 분포" — 각 점 $x$마다 $f(x)$가 Gaussian, 여러 점은 multivariate Gaussian.

- **Prior** $f \sim \mathcal{GP}(m, k)$: $m(x)$ 평균, $k(x, x')$ covariance (kernel)
- **Posterior** $f | D \sim \mathcal{GP}(\mu_n, k_n)$: 닫힌형

### BO 사이클

```
1. Initial design: evaluate f at x_1, ..., x_m
2. Fit GP posterior on data
3. Compute acquisition a(x) (EI, UCB, ...)
4. x_next = argmax a(x)
5. Evaluate y_next = f(x_next)
6. Add to data, repeat from 2
```

### 왜 효율적

각 step에서 **posterior uncertainty를 이용**하여 "아직 탐색 안 된 + 유망한" 영역 선택. Random search / grid보다 10~1000× 적은 evaluations.

### 요리 비유

"블랙박스 요리사의 실력 평가":
- 몇 번 시식 → 어떤 메뉴에서 잘할지 **확률 분포**
- 다음엔 "가장 좋을 것 같은 + 아직 안 먹어본" 메뉴 선택

---

## ✏️ 엄밀한 정의

### 정의 1.1 — Gaussian Process

확률과정 $f: \mathcal{X} \to \mathbb{R}$가 **Gaussian Process**라는 것:

모든 유한 $\{x_1, \ldots, x_n\}$에 대해 $(f(x_1), \ldots, f(x_n))$이 multivariate Gaussian:

$$f \sim \mathcal{GP}(m, k) \iff (f(x_i))_i \sim \mathcal{N}(m, K), K_{ij} = k(x_i, x_j)$$

- **Mean function** $m(x)$
- **Kernel** (covariance function) $k(x, x')$

### 정의 1.2 — 주요 Kernel

**Squared Exponential (RBF)**:
$$k(x, x') = \sigma_f^2 \exp\left(-\frac{\|x-x'\|^2}{2\ell^2}\right)$$

**Matérn-3/2, 5/2**: less smooth alternatives.

**Linear**: $k(x, x') = x^T x'$ (Bayesian linear regression).

### 정의 1.3 — GP Posterior (Conditioning)

Observations $y_i = f(x_i) + \epsilon_i, \epsilon_i \sim \mathcal{N}(0, \sigma_n^2)$.

$$\mu_n(x) = k(x, X)[K + \sigma_n^2 I]^{-1}y$$

$$\sigma_n^2(x) = k(x, x) - k(x, X)[K + \sigma_n^2 I]^{-1}k(X, x)$$

$K \in \mathbb{R}^{n \times n}, k(x, X) \in \mathbb{R}^{1 \times n}$.

### 정의 1.4 — Acquisition Function

$$x_{n+1} = \arg\max_{x \in \mathcal{X}} a(x; \mathcal{D}_n)$$

$a$는 GP posterior에 기반. 예: EI, UCB, TS, PI (Ch6-02).

---

## 🔬 정리와 증명

### 정리 1.1 — GP Posterior = Conditional Gaussian

**명제**: $f \sim \mathcal{GP}(0, k)$, observations $y = f(X) + \epsilon, \epsilon\sim\mathcal{N}(0, \sigma_n^2 I)$이면:

$$f(x_*)|y \sim \mathcal{N}(\mu_*(x_*), \sigma_*^2(x_*))$$

with $\mu_*, \sigma_*^2$ as above.

**증명**:

Joint $(f_*, y)^T \sim \mathcal{N}(0, \Sigma)$:
$$\Sigma = \begin{pmatrix}k(x_*, x_*) & k(x_*, X)\\ k(X, x_*) & K + \sigma_n^2 I\end{pmatrix}$$

Conditional formula:
$$f_*|y \sim \mathcal{N}(k_*^T(K+\sigma_n^2 I)^{-1}y, k_{**} - k_*^T(K+\sigma_n^2 I)^{-1}k_*)$$

$\square$

### 정리 1.2 — GP Posterior Mean 해석

**명제**: Posterior mean $\mu_n(x)$는 **training point의 kernel-weighted sum**:

$$\mu_n(x) = \sum_i \alpha_i k(x, x_i), \quad \alpha = (K + \sigma_n^2 I)^{-1}y$$

**귀결**: GP regression은 **kernel ridge regression**과 같음 (MAP view).

### 정리 1.3 — GP Posterior Variance의 감소 property

**명제**: New observation $(x_{n+1}, y_{n+1})$ 추가 시:

$$\sigma_{n+1}^2(x) \leq \sigma_n^2(x) \quad \forall x$$

(Gaussian conditioning의 variance reduction).

**증명**: Schur complement. $\sigma^2_{n+1}(x) = \sigma_n^2(x) - \text{Cov}_n(x, x_{n+1})^2/(\sigma_n^2(x_{n+1}) + \sigma_n^2)$. $\geq 0$ subtraction. $\square$

**BO에 중요**: 관측할수록 posterior 좁아짐 → exploration 줄고 exploitation 증가.

### 정리 1.4 — Kernel Hyperparameter Learning

**명제**: Kernel의 hyperparams ($\ell, \sigma_f, \sigma_n$)를 **marginal likelihood 최대화**:

$$\log p(y|X; \theta_k) = -\frac{1}{2}y^T K_y^{-1} y - \frac{1}{2}\log|K_y| - \frac{n}{2}\log(2\pi)$$

$K_y = K + \sigma_n^2 I$.

**증명**: Gaussian marginal likelihood 직접 유도. $\square$

**실전**: GPy/GPyTorch/BoTorch가 L-BFGS로 $\theta_k$ 최적화.

### 정리 1.5 — 계산 복잡도

**명제**: 
- Posterior fit: $O(n^3)$ (matrix inversion)
- Prediction at 1 new point: $O(n^2)$
- Acquisition 최적화: $O(n^2 \cdot |\mathcal{X}_{\text{candidate}}|)$

$n > 1000$에서 비용이 폭발 → sparse GP, inducing points (Titsias 2009) 필요.

---

## 💻 GPyTorch + BoTorch 구현

```python
import torch
import gpytorch
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(0)

# Black-box function (unknown in real application)
def f_true(x):
    return -(x * torch.sin(x) + 0.1 * torch.cos(5*x))

# ────────────────────────────────────────────────
# Initial data
# ────────────────────────────────────────────────
train_x = torch.linspace(0, 10, 5).unsqueeze(1)
train_y = f_true(train_x.squeeze()).unsqueeze(0).squeeze()

# ────────────────────────────────────────────────
# GP Model
# ────────────────────────────────────────────────
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x))

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x.squeeze(), train_y, likelihood)

# Fit
model.train(); likelihood.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
for _ in range(100):
    optimizer.zero_grad()
    output = model(train_x.squeeze())
    loss = -mll(output, train_y)
    loss.backward(); optimizer.step()

# ────────────────────────────────────────────────
# Posterior predictive + acquisition (EI)
# ────────────────────────────────────────────────
model.eval(); likelihood.eval()
x_test = torch.linspace(0, 10, 400).unsqueeze(1)
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    post = likelihood(model(x_test.squeeze()))
    mean = post.mean.numpy()
    lower, upper = post.confidence_region()
    lower, upper = lower.numpy(), upper.numpy()

# EI (will detail in Ch6-02)
y_best = train_y.min().item()  # minimization
mu = mean
sigma = (upper - lower) / 4  # 95% CI → ±2σ approx
from scipy.stats import norm
z = (y_best - mu) / (sigma + 1e-9)
ei = (y_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)

fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
x_plot = x_test.numpy().flatten()

axes[0].fill_between(x_plot, lower, upper, alpha=0.3, label='95% CI')
axes[0].plot(x_plot, mean, 'b-', lw=2, label='GP mean')
axes[0].plot(x_plot, f_true(torch.tensor(x_plot)).numpy(), 'k--', alpha=0.5, label='true f (unknown in reality)')
axes[0].scatter(train_x.numpy(), train_y.numpy(), c='red', s=50, zorder=5, label='data')
axes[0].set_ylabel('f(x)'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title('GP posterior after 5 observations')

axes[1].plot(x_plot, ei, 'g-', lw=2, label='EI acquisition')
axes[1].axvline(x_plot[ei.argmax()], color='r', ls='--', label=f'next x = {x_plot[ei.argmax()]:.2f}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('EI'); axes[1].legend(); axes[1].grid(alpha=0.3)

plt.tight_layout(); plt.savefig('gp_bo.png', dpi=150); plt.show()
```

---

## 🔗 AI/ML 연결

### Hyperparameter Tuning
BoTorch + Optuna/Ax: 딥러닝 LR·width·depth 조율.

### Neural Architecture Search (NAS)
BO over architecture choices (Bergstra 2013, Zoph 2017).

### Drug Discovery
분자 property optimization with GP surrogate (Zhavoronkov 2019).

### A/B Testing
Bayesian Sequential Design for conversion rate optimization.

### Robotics
Policy parameter tuning with BO (Martinez-Cantin 2007).

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| GP prior 적절 | Wrong kernel/hyperparams → poor posterior |
| Continuous $\mathcal{X}$ | Discrete·categorical 필요 시 custom kernel |
| Smooth $f$ | Step function·noise heavy-tailed 어려움 |
| $n < 1000$ | Large $n$ → sparse GP, inducing points |
| Low dim | High-dim BO 어려움 (Ch6-04) |

---

## 📌 핵심 정리

$$\boxed{f|y \sim \mathcal{GP}(\mu_n, k_n), \quad \mu_n(x) = k_*^T(K+\sigma^2 I)^{-1}y}$$

BO cycle:
1. GP posterior 계산
2. Acquisition $a(x)$ → $x_{\text{next}}$
3. $f$ evaluation → 데이터 추가
4. Repeat

---

## 🤔 생각해볼 문제

**문제 1** (기초): GP kernel의 length-scale $\ell$이 의미하는 것은?

<details>
<summary>해설</summary>

$\ell$: "**correlation distance**". $k(x, x') = \exp(-\|x-x'\|^2/(2\ell^2))$.

- $\ell$ 작음: $x, x'$가 가까워도 빠르게 uncorrelated → wiggly posterior
- $\ell$ 큼: long-range correlation → smooth posterior

**BO에 중요**: 
- $\ell$ 작으면 → 각 관측이 local 영향만 → exploration 많이 필요
- $\ell$ 크면 → 관측이 globally 전파 → 빠른 수렴

Auto-ML: marginal likelihood로 $\ell$ 학습 (정리 1.4).

</details>

**문제 2** (심화): GP에 noise $\sigma_n^2$이 있을 때 다른 $x$에서 관측하는 것과 **같은 $x$에서 반복 관측**의 posterior 효과?

<details>
<summary>해설</summary>

**다른 $x$**: 새로운 위치에서 정보 → posterior가 **공간적으로 확장** + narrow.

**같은 $x$ 반복** ($n$번): $y_i = f(x) + \epsilon_i$, $\epsilon_i$ 독립. $\hat f(x) = \bar y$의 noise $\sigma_n/\sqrt n$ → 점점 정확.

Effective:
- Other $x$: broader info, posterior narrowing at **all** nearby points (via kernel correlation)
- Same $x$: posterior at **only that $x$** (and correlated points)

**Optimal strategy**: BO는 보통 new $x$ 선호 (acquisition이 uncertainty reward). 하지만 **high-noise region**에서는 replication 가치.

**Multi-fidelity BO** (Ch6-04): high-noise/cheap evaluation + 간혹 expensive noiseless.

</details>

**문제 3** (AI 연결): BO vs Grid/Random Search vs HyperBand 비교?

<details>
<summary>해설</summary>

| | Grid | Random | BO | HyperBand |
|---|---|---|---|---|
| Samples needed | $O(d^k)$ | $O(n)$ uniformly | $O(\log T)$ near-optimal | Early stopping savings |
| Use info | No | No | **Yes (GP)** | Partial |
| Parallel | Easy | Easy | Batch BO 복잡 | Natural |
| Discrete | OK | OK | Custom kernel | OK |
| High-dim | Fails | Better | Hard | OK |

**실전**:
- $d < 20$, expensive evaluation: **BO** 최적
- $d > 20$, cheap: **Random** + HyperBand
- **BOHB** (Falkner 2018) = BO + HyperBand 결합 — 최근 표준

Hugging Face, WandB, Ax 등의 현대 HPO tool 모두 BO 기반.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch5-05 SWAG와 SGLD](../ch5-bayesian-nn/05-swag-sgld.md) | [📚 README로 돌아가기](../README.md) | [02. Acquisition Function들 ▶](./02-acquisition-functions.md) |

</div>
