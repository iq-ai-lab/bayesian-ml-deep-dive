# 02. Probabilistic Programming 언어

## 🎯 핵심 질문

- **Stan / PyMC / NumPyro / Pyro** 각 PPL의 철학과 backend 차이?
- **ADVI** (Automatic Differentiation VI)는 Stan/PyMC에서 어떻게 자동 VI를 제공하는가?
- 모델 코드 → posterior 추론까지의 **추상화 과정**은?
- 언제 어느 PPL을 쓸지의 실전 기준?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**PPL이 없었다면 Bayesian workflow는 수학+코드 모두 직접 짜야 함**. PPL이 Bayesian을 mainstream으로 — "**모델만 작성 → posterior 얻음**". Stan이 clinical trials에서 사용되고 PyMC가 industry data science에 침투한 것은 PPL의 승리. NumPyro·Pyro는 **딥러닝과 Bayesian 결합**의 현대 interface.

---

## 📐 수학적 선행 조건

- [Ch2-01~06](../ch2-variational-inference/01-vi-idea-elbo.md): VI
- [Ch4 전체](../ch4-mcmc/01-metropolis-hastings.md): MCMC
- Auto-diff basics

---

## 📖 직관적 이해

### PPL의 공통 철학

"**모델 = 확률변수 간 관계** → 자동 posterior 추론". 사용자는:
1. **Prior** 명시
2. **Likelihood** 명시 (observed data)
3. 추론 (NUTS, VI, ...)

Framework이 gradient, posterior sampling, diagnostics 자동 처리.

### 4대 PPL 비교

| | Stan | PyMC | NumPyro | Pyro |
|---|---|---|---|---|
| Backend | C++ | PyTensor | JAX | PyTorch |
| 주 사용자 | 통계학자 | Data scientist | ML researcher | Deep learning |
| NUTS | ✓ (원조) | ✓ | ✓ (빠름) | ✓ |
| ADVI | ✓ | ✓ | ✓ | ✓ (SVI) |
| GPU | 제한적 | 부분적 | **완전 (JAX)** | **완전 (PyTorch)** |
| Neural nets | 제한적 | 가능 | **자연** | **자연** |
| Discrete params | 자동 | 자동 | 수동 marginalization | 수동 |

### 요리 비유

"**레시피 선언**만 하면 요리가 완성되는 AI 부엌":
- Stan: 통계학자의 주방 (정밀하지만 전문)
- PyMC: 가정용 (user-friendly, Python-native)
- NumPyro/Pyro: 딥러닝 shop과 연계된 주방

---

## ✏️ 엄밀한 정의

### 정의 2.1 — PPL

**Probabilistic Programming Language**: 사용자가 확률모형을 code로 작성 → compiler/runtime이 inference engine 호출:

```
model:
    prior: W ~ Normal(0, 1)
    likelihood: y | W, x ~ Normal(W*x, σ)
infer:
    sample(model, data) or vi(model, data)
```

### 정의 2.2 — ADVI (Automatic Differentiation VI)

**Kucukelbir et al. 2017** — Stan의 auto VI:

1. Variable unconstrain → $\mathbb{R}^d$ (logit, log, etc.)
2. Mean-field Gaussian on unconstrained: $q(\theta) = \mathcal{N}(\mu, \text{diag}(\sigma^2))$
3. Reparam (Ch2-05) + ELBO gradient + SGD

모든 PPL이 유사 기법 채택.

### 정의 2.3 — Universal vs Restricted PPL

- **Universal**: arbitrary Python code (Pyro, NumPyro) — control flow 포함
- **Restricted**: declarative model only (Stan) — auto-optimization 용이

---

## 🔬 각 PPL의 특징

### Stan

```stan
data {
  int N;
  vector[N] x;
  vector[N] y;
}
parameters {
  real alpha;
  real beta;
  real<lower=0> sigma;
}
model {
  alpha ~ normal(0, 10);
  beta ~ normal(0, 1);
  sigma ~ exponential(1);
  y ~ normal(alpha + beta * x, sigma);
}
```

- Compiled to C++ → **fast NUTS**
- Standard for clinical trials, econometrics
- Posterior predictive check built-in
- `cmdstanpy`, `rstan` for Python/R interfaces

### PyMC (Python)

```python
import pymc as pm

with pm.Model() as model:
    alpha = pm.Normal('alpha', 0, 10)
    beta = pm.Normal('beta', 0, 1)
    sigma = pm.Exponential('sigma', 1)
    y = pm.Normal('y', mu=alpha + beta * x, sigma=sigma, observed=y_obs)
    
    trace = pm.sample(2000, tune=1000, target_accept=0.9)
```

- PyTensor backend
- Pythonic, integrates with numpy
- ArviZ for diagnostics
- GPU via PyTensor (limited)

### NumPyro (JAX)

```python
import numpyro
import numpyro.distributions as dist
import jax.numpy as jnp

def model(x, y=None):
    alpha = numpyro.sample('alpha', dist.Normal(0, 10))
    beta = numpyro.sample('beta', dist.Normal(0, 1))
    sigma = numpyro.sample('sigma', dist.Exponential(1))
    numpyro.sample('obs', dist.Normal(alpha + beta * x, sigma), obs=y)

from numpyro.infer import MCMC, NUTS
kernel = NUTS(model)
mcmc = MCMC(kernel, num_warmup=1000, num_samples=2000)
mcmc.run(rng_key, x=x, y=y_obs)
```

- JAX backend → **매우 빠름** (GPU/TPU)
- Functional style
- Direct integration with JAX ML ecosystem

### Pyro (PyTorch)

```python
import pyro
import pyro.distributions as dist

def model(x, y=None):
    alpha = pyro.sample('alpha', dist.Normal(0, 10))
    beta = pyro.sample('beta', dist.Normal(0, 1))
    sigma = pyro.sample('sigma', dist.Exponential(1))
    with pyro.plate('data', len(x)):
        pyro.sample('obs', dist.Normal(alpha + beta * x, sigma), obs=y)

from pyro.infer.autoguide import AutoNormal
guide = AutoNormal(model)
from pyro.infer import SVI, Trace_ELBO
svi = SVI(model, guide, torch.optim.Adam({'lr': 1e-3}), loss=Trace_ELBO())
for step in range(5000):
    svi.step(x, y_obs)
```

- PyTorch backend
- Deep probabilistic programming (VAE, BNN 자연)
- SVI, HMC, advanced inference patterns

### 실전 가이드

**선택 흐름**:
- 통계학자, clinical trial, 재현 중요: **Stan**
- Data scientist, business analytics: **PyMC**
- ML researcher, GPU-heavy, Bayesian DL: **NumPyro** or **Pyro**
- Deep probabilistic (VAE in Bayesian framework): **Pyro**

---

## 💻 공통 예제 — Hierarchical Linear Regression

```python
import numpy as np
import pymc as pm

# Simulated multi-group data
np.random.seed(0)
n_groups = 8
group_idx = np.repeat(np.arange(n_groups), 30)
x = np.random.randn(len(group_idx))
beta_true = np.random.normal(1.0, 0.3, n_groups)
y = beta_true[group_idx] * x + 0.5 * np.random.randn(len(x))

with pm.Model() as model:
    mu_b = pm.Normal('mu_b', 0, 5)
    sd_b = pm.HalfNormal('sd_b', 1)
    beta = pm.Normal('beta', mu_b, sd_b, shape=n_groups)
    sigma = pm.HalfNormal('sigma', 1)
    pm.Normal('y', beta[group_idx]*x, sigma, observed=y)
    
    # Two inference options
    trace_nuts = pm.sample(2000, tune=1000, random_seed=0)
    approx = pm.fit(20000, method='advi')  # VI
    trace_advi = approx.sample(2000)

# Compare
import arviz as az
print("NUTS:")
print(az.summary(trace_nuts, var_names=['mu_b', 'sd_b']))
print("\nADVI:")
print(az.summary(trace_advi, var_names=['mu_b', 'sd_b']))
```

---

## 🔗 AI/ML 연결

### Pyro VAE
Pyro로 VAE 자연 표현 — encoder = guide, decoder = model.

### Stan in Drug Discovery
Pharmacokinetic/pharmacodynamic modeling.

### NumPyro + HMC on GPU
수만 데이터의 large-scale Bayesian hierarchical analysis.

### Edward2 / TensorFlow Probability
Google의 PPL (Stan-inspired, TF backend).

### BUGS/JAGS (Legacy)
초기 PPL. Academic tutorial에 흔.

---

## ⚖️ 가정과 한계

| PPL | 장점 | 단점 |
|-----|------|------|
| Stan | 성숙, 통계 정확 | Python integration 간접 |
| PyMC | Pythonic, 쉬움 | 느림 (PyTensor overhead) |
| NumPyro | 매우 빠름 (JAX) | Learning curve, JAX 생태계 종속 |
| Pyro | Deep models 자연 | NUTS는 slower than NumPyro |

**공통 한계**:
- Discrete parameter inference 어려움 (automatic NUTS 불가)
- Posterior geometry: funnel/pathological은 여전히 문제
- Scalability: 수백만 params 힘듦

---

## 📌 핵심 정리

| PPL | Backend | 주요 용도 |
|-----|---------|-----------|
| **Stan** | C++ | 표준 통계 분석, clinical |
| **PyMC** | PyTensor | Data science, 비즈니스 |
| **NumPyro** | JAX | 대규모 Bayesian, GPU |
| **Pyro** | PyTorch | Deep probabilistic (VAE, BNN) |

공통: NUTS + ADVI 자동 제공. Model code → posterior sample.

---

## 🤔 생각해볼 문제

**문제 1** (기초): Stan code에서 `<lower=0>` 같은 constraint는 왜 필요?

<details>
<summary>해설</summary>

Stan이 **unconstrained space에서 NUTS**를 실행하기 위함:
- `<lower=0>` → $\log$ 변환 (positive real)
- `<lower=0, upper=1>` → logit 변환 (probability)
- Simplex → stick-breaking

자동 Jacobian adjustment 포함 → correct posterior.

없이 쓰면 NUTS가 invalid region으로 proposal → divergence.

PyMC·Pyro·NumPyro도 내부적으로 비슷 (bijectors).

</details>

**문제 2** (심화): ADVI가 **전역 최적** 보장 안 하는 이유는?

<details>
<summary>해설</summary>

ELBO는 **비볼록** in $\phi$ (특히 NN-parameterized). SGD → **local optimum**.

Multimodal posterior → VI가 한 mode에 갇힘 (Ch2-01 reverse KL property).

실전 완화:
- **Multiple restarts** (different init)
- **Ensemble of approximations**
- **Normalizing flow** as $q$ (Ch3-03)

**Pathological geometry** (funnel): ADVI fails → reparameterization 필요 (non-centered parameterization).

</details>

**문제 3** (AI 연결): PyTorch model + Pyro VI로 BNN을 구현할 때의 typical workflow?

<details>
<summary>해설</summary>

```python
import pyro
import pyro.nn as pnn
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.autoguide import AutoDiagonalNormal

class BNN(pnn.PyroModule):
    def __init__(self):
        super().__init__()
        self.fc1 = pnn.PyroModule[torch.nn.Linear](1, 50)
        self.fc1.weight = pnn.PyroSample(dist.Normal(0, 1).expand([50, 1]))
        self.fc1.bias = pnn.PyroSample(dist.Normal(0, 1).expand([50]))
        self.fc2 = pnn.PyroModule[torch.nn.Linear](50, 1)
        self.fc2.weight = pnn.PyroSample(dist.Normal(0, 1).expand([1, 50]))
        self.fc2.bias = pnn.PyroSample(dist.Normal(0, 1).expand([1]))
    def forward(self, x, y=None):
        h = torch.tanh(self.fc1(x))
        mean = self.fc2(h).squeeze(-1)
        with pyro.plate('data', x.shape[0]):
            obs = pyro.sample('obs', dist.Normal(mean, 0.1), obs=y)
        return mean

model = BNN()
guide = AutoDiagonalNormal(model)  # automatic mean-field Gaussian VI
svi = SVI(model, guide, optim, Trace_ELBO())
```

`AutoDiagonalNormal`이 Bayes by Backprop equivalent. Prior와 likelihood만 declare하면 나머지 자동.

**장점**: Pyro code = "model declaration" — inference engine이 나머지.

Discrete, normalizing flow, structured posterior 모두 customizable.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. Diffusion Model의 Bayesian 해석](./01-diffusion-bayesian.md) | [📚 README로 돌아가기](../README.md) | [03. Bayesian Deep Learning의 불확실성 분해 ▶](./03-epistemic-aleatoric.md) |

</div>
