# 03. Bayesian Deep Learning의 불확실성 분해

## 🎯 핵심 질문

- 분해 공식 $\text{Var}[y^*] = \mathbb{E}_W[\text{Var}(y^*|x^*, W)] + \text{Var}_W[\mathbb{E}(y^*|x^*, W)]$은 어떻게 유도되는가?
- **Epistemic**(모델 불확실성, 데이터로 감소)과 **aleatoric**(관측 노이즈, 감소 불가)의 본질적 차이?
- 각각을 수학적으로 어떻게 분리하고 실전에서 어떻게 추정?
- 언제 어느 uncertainty가 중요한가 (active learning vs safety)?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

"**모델이 뭘 모르는가**"(epistemic) vs "**세상이 본질적으로 랜덤**"(aleatoric) — 구분이 **critical**한 응용 많음. 의료 진단에서 "환자 정보 부족" vs "같은 증상에도 다양한 outcome". 자율주행의 OOD detection, active learning의 acquisition, reliable confidence scoring — 모두 이 구분에 의존. Kendall & Gal (2017)의 "What uncertainties do we need"가 이 framework의 decisive paper.

---

## 📐 수학적 선행 조건

- [Ch1-04 Predictive](../ch1-bayesian-foundation/04-predictive-distribution.md): law of total variance
- [Ch5 전체](../ch5-bayesian-nn/01-bnn-formulation.md): BNN
- Heteroscedastic 회귀 기초

---

## 📖 직관적 이해

### 두 불확실성의 구분

**Epistemic (모델 불확실성)**:
- 어떤 가중치가 맞는지 모름
- **Data 많아지면 0으로 감소**
- "Reducible uncertainty"

**Aleatoric (데이터 불확실성)**:
- 같은 $x$에서도 $y$가 다양 (noise, inherent stochasticity)
- **Data 많아져도 감소 안 함**
- "Irreducible uncertainty"

### 요리 비유

- **Epistemic**: "요리사에 대해 모름 (who is cooking?)" — 더 관찰하면 알게 됨
- **Aleatoric**: "같은 요리사도 매일 조금 다르게 요리" — 본질적 다양성

### 분해의 수학적 구조

Bayesian total variance:
$$\text{Var}[y^*|x^*, D] = \mathbb{E}_{W|D}[\text{Var}(y^*|x^*, W)] + \text{Var}_{W|D}[\mathbb{E}(y^*|x^*, W)]$$

- 첫 항: 각 $W$에서 의 내재적 noise → **aleatoric**
- 둘째 항: $W$의 posterior variation → **epistemic**

### 실전 추정

BNN with $T$ MC samples of $W^{(t)} \sim q(W|D)$:

$$\text{Aleatoric} \approx \frac{1}{T}\sum_t \sigma_{W^{(t)}}^2(x^*)$$

$$\text{Epistemic} \approx \frac{1}{T}\sum_t \mu_{W^{(t)}}(x^*)^2 - \left(\frac{1}{T}\sum_t \mu_{W^{(t)}}(x^*)\right)^2$$

NN이 **$(\mu(x), \sigma(x))$ 둘 다 output**하면 heteroscedastic + aleatoric 분리 가능.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Predictive Mean & Variance

BNN with posterior $p(W|D)$:

$$\mathbb{E}[y^*|x^*, D] = \int \mathbb{E}[y^*|x^*, W] p(W|D) dW$$

$$\text{Var}[y^*|x^*, D] = \text{Var}_{\text{total}}$$

### 정의 3.2 — Total Variance Decomposition

$$\boxed{\text{Var}[y^*|x^*, D] = \underbrace{\mathbb{E}_{W|D}[\text{Var}(y^*|x^*, W)]}_{\sigma_{\text{ale}}^2(x^*)} + \underbrace{\text{Var}_{W|D}[\mathbb{E}(y^*|x^*, W)]}_{\sigma_{\text{epi}}^2(x^*)}}$$

### 정의 3.3 — Heteroscedastic Aleatoric

Model이 **$x$-dependent noise** 예측:

$$p(y|x, W) = \mathcal{N}(\mu_W(x), \sigma_W^2(x))$$

$\sigma_W(x)$ = NN output. Aleatoric이 $x$마다 다름.

### 정의 3.4 — Homoscedastic Aleatoric

$\sigma$ = constant (data-independent). 더 간단한 모델.

---

## 🔬 정리와 증명

### 정리 3.1 — Total Variance Law (Ch1-04 정리 4.1 재정리)

**명제**:
$$\text{Var}[y^*|x^*, D] = \mathbb{E}_{W}[\text{Var}(y^*|x^*, W)] + \text{Var}_{W}[\mathbb{E}(y^*|x^*, W)]$$

**증명**: Ch1-04 정리 4.1. $\square$

### 정리 3.2 — 데이터 증가와 Epistemic 감소

**명제**: $N \to \infty$에서 Bernstein-von Mises (Ch1-05 정리 5.1) 조건 하:

$$p(W|D_N) \xrightarrow{TV} \mathcal{N}(W^*, F^{-1}/N)$$

따라서:
$$\text{Var}_{W|D}[\mathbb{E}(y^*|x^*, W)] = O(1/N) \to 0$$

**증명 sketch**: Posterior 분산이 $O(1/N)$ → NN output $\mathbb{E}(y^*|x^*, W)$의 variance도 $O(1/N)$ (delta method). $\square$

**귀결**: Epistemic이 **reducible**의 수학적 증명.

### 정리 3.3 — Aleatoric의 불변성

**명제**: $\sigma_{\text{ale}}^2(x^*) \to \sigma_{W^*}^2(x^*)$ (true parameter's noise) in large-$N$ limit.

**증명**: 첫 항 $\mathbb{E}_W[\text{Var}(y|W)]$가 $N\to\infty$에서 delta posterior에 의해 $\text{Var}(y|W^*)$로 수렴. 이것이 model-specified noise. $\square$

**귀결**: Aleatoric이 **irreducible**.

### 정리 3.4 — Mutual Information Decomposition

**명제** (Houlsby et al. 2011 BALD):

$$H[y^*|x^*, D] = \underbrace{\mathbb{E}_{W}[H(y^*|x^*, W)]}_{\text{aleatoric entropy}} + \underbrace{I(y^*, W | x^*, D)}_{\text{epistemic info}}$$

Total predictive entropy = aleatoric + mutual info between $y^*$ and $W$.

**증명**: Chain rule of entropy. $\square$

**BALD**: Active learning acquisition = $I(y^*, W|x^*, D)$ — epistemic을 max하는 $x$ 선택.

### 정리 3.5 — Classification의 경우

**Softmax BNN**: $p(y = c|x^*, W) = \text{softmax}(f_W(x^*))_c$.

**Predictive**: $\bar p_c = \frac{1}{T}\sum_t p_c^{(t)}$.

- **Predictive entropy**: $H[\bar p] = -\sum_c \bar p_c \log \bar p_c$
- **Average entropy**: $\frac{1}{T}\sum_t H[p^{(t)}]$
- **Epistemic** (mutual info): $H[\bar p] - \frac{1}{T}\sum_t H[p^{(t)}]$

BALD acquisition = epistemic. Higher → more worth labeling.

---

## 💻 PyTorch 구현 — Heteroscedastic BNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

class HeteroBNN(nn.Module):
    """MC Dropout BNN that outputs (μ, log σ²) per x."""
    def __init__(self, hidden=50, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(1, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc_mu = nn.Linear(hidden, 1)
        self.fc_logvar = nn.Linear(hidden, 1)
        self.dropout = dropout
    def forward(self, x):
        h = F.dropout(F.relu(self.fc1(x)), p=self.dropout, training=True)
        h = F.dropout(F.relu(self.fc2(h)), p=self.dropout, training=True)
        return self.fc_mu(h), self.fc_logvar(h)

# Data with heteroscedastic noise
N = 100
x = torch.linspace(-3, 3, N).unsqueeze(1)
noise_std = 0.05 + 0.3*torch.abs(x)  # x가 클수록 noise 증가
y = torch.sin(x) + noise_std * torch.randn(N, 1)

model = HeteroBNN(hidden=100, dropout=0.1)
opt = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=1e-4)

# Negative log-likelihood with heteroscedastic Gaussian
def nll_loss(mu, logvar, y):
    return 0.5 * (logvar + (y - mu)**2 / torch.exp(logvar)).mean()

for epoch in range(3000):
    opt.zero_grad()
    mu, logvar = model(x)
    loss = nll_loss(mu, logvar, y)
    loss.backward(); opt.step()

# MC Dropout predictive
x_test = torch.linspace(-5, 5, 200).unsqueeze(1)
T = 200
mus, logvars = [], []
with torch.no_grad():
    for _ in range(T):
        mu, lv = model(x_test)
        mus.append(mu.numpy().flatten())
        logvars.append(lv.numpy().flatten())
mus = np.array(mus); logvars = np.array(logvars)

# Decomposition
mean_pred = mus.mean(axis=0)
ale_var = np.exp(logvars).mean(axis=0)  # E_W[σ²] = aleatoric
epi_var = mus.var(axis=0)                # Var_W[μ] = epistemic
total_var = ale_var + epi_var

fig, ax = plt.subplots(figsize=(12, 6))
x_plot = x_test.numpy().flatten()
ax.plot(x_plot, mean_pred, 'b-', lw=2, label='Posterior mean')
ax.fill_between(x_plot, mean_pred - 2*np.sqrt(ale_var), mean_pred + 2*np.sqrt(ale_var),
                alpha=0.3, color='green', label='Aleatoric ±2σ')
ax.fill_between(x_plot, mean_pred - 2*np.sqrt(total_var), mean_pred + 2*np.sqrt(total_var),
                alpha=0.2, color='red', label='Total ±2σ (aleatoric + epistemic)')
ax.scatter(x.numpy(), y.numpy(), s=15, color='black', alpha=0.5, label='data')
ax.plot(x_plot, np.sin(x_plot), 'k--', alpha=0.5, label='true f')
ax.legend(); ax.grid(alpha=0.3)
ax.set_title('Heteroscedastic BNN — aleatoric (noise) vs epistemic (OOD) 분리')
plt.tight_layout(); plt.savefig('epi_ale.png', dpi=150); plt.show()

print(f"Center (x=0): aleatoric={np.sqrt(ale_var[100]):.3f}, epistemic={np.sqrt(epi_var[100]):.3f}")
print(f"OOD (x=5):    aleatoric={np.sqrt(ale_var[-1]):.3f}, epistemic={np.sqrt(epi_var[-1]):.3f}")
```

**기대 결과**:
- 데이터 중심: aleatoric 작 (low noise), epistemic 작음 (well-observed)
- 데이터 가장자리 (큰 $|x|$): aleatoric 커짐 (heteroscedastic), epistemic 폭증 (OOD)
- **두 uncertainty의 source가 다름**이 그래프에 분리 표시

---

## 🔗 AI/ML 연결

### Active Learning
BALD = epistemic mutual info → acquisition.

### OOD Detection
High epistemic → "모델이 이 입력에 대해 잘 모름" → OOD 의심.

### Medical AI
Aleatoric high: "본질적 불확실성 — doctor 2차 의견"
Epistemic high: "training data 부족 — 유사 사례 찾아야"

### Autonomous Driving
Aleatoric: "날씨·센서 noise"
Epistemic: "never-seen environment" → 주의

### Reinforcement Learning
Epistemic-based exploration — "uncertain state 먼저 explore".

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Posterior $p(W\|D)$ 잘 근사 | Poor BNN → wrong decomposition |
| Heteroscedastic model | Data가 충분해야 $\sigma(x)$ 학습 |
| Gaussian likelihood | Non-Gaussian (e.g., Poisson) 에선 공식 다름 |
| Model well-specified | Misspecification 시 분해 해석 부정확 |

**실전 팁**:
- **Homoscedastic은 baseline**. Heteroscedastic로 upgrade
- **Deep ensembles**도 epistemic 추정 가능 (MC Dropout 대안)
- **Temperature scaling** (Ch7-04)와 별개 — uncertainty의 **magnitude** vs **calibration**

---

## 📌 핵심 정리

$$\boxed{\text{Var}[y^*|x^*, D] = \underbrace{\mathbb{E}_W[\text{Var}(y^*|W)]}_{\text{aleatoric, irreducible}} + \underbrace{\text{Var}_W[\mathbb{E}(y^*|W)]}_{\text{epistemic, reducible}}}$$

- **Epistemic**: $\propto 1/N$, reducible by more data
- **Aleatoric**: 데이터로 감소 안 함 (inherent noise)
- **실전 추정**: BNN $T$ samples → per-$x$ decomposition
- **응용**: active learning, OOD detection, safety-critical decisions

---

## 🤔 생각해볼 문제

**문제 1** (기초): 동전 던지기에서 $\theta$의 posterior를 Bayesian으로 추정. 다음 던지기에서의 aleatoric vs epistemic은?

<details>
<summary>해설</summary>

$p(y = 1 | \theta) = \theta$ (Bernoulli).

- **Aleatoric** (given $\theta$): $\text{Var}[y|\theta] = \theta(1-\theta)$. $\theta = 0.5$에서 최대 (inherent randomness).
- **Epistemic**: $\text{Var}_\theta[\mathbb{E}(y|\theta)] = \text{Var}_\theta[\theta] = \text{Var}[\theta|D]$ → posterior variance.

$N \to \infty$에서 posterior → delta → epistemic → 0. Aleatoric → $\theta_0(1-\theta_0)$ 여전.

**의미**: 동전 자체의 randomness는 학습 데이터로 없앨 수 없음.

</details>

**문제 2** (심화): MC Dropout의 epistemic이 "진짜 epistemic이 아니다"라는 비판 (Ch5-04 참조). 이 framework에서 재검토하면?

<details>
<summary>해설</summary>

Osband 비판: MC Dropout variance가 $N \to \infty$에서도 작아지지 않음 (dropout rate $p$에 의존).

**Our framework**: 진짜 epistemic은 $O(1/N)$ (정리 3.2). MC Dropout은:
- Bernoulli posterior family의 제약
- Dropout rate $p$가 prior strength 결정
- $N$과 무관한 constant bias

**결론**: MC Dropout의 "epistemic"은 approximate — 실전 잘 동작하지만 **theoretical reducibility 보장 없음**.

**더 정확한 epistemic**:
- Laplace (Ch5-02): BvM 하에서 $1/N$
- Deep ensembles: multi-mode 탐색
- Full HMC (가능한 경우): exact

**Practical advice**: OOD detection 같이 "상대적 크기" 중요하면 MC Dropout OK. "절대적 reducible estimate" 필요하면 다른 방법.

</details>

**문제 3** (AI 연결): LLM의 hallucination을 이 framework로 설명하면?

<details>
<summary>해설</summary>

**가설**: Hallucination은 **epistemic uncertainty가 높은 영역에서 모델이 과신**.

**Aleatoric**: 자연어 자체의 다양성 (paraphrase, 동의어)
**Epistemic**: 모델이 본 적 없는 facts (training data 밖)

**관찰**:
- Factual hallucination: high epistemic (training knowledge 부족)
- Creative generation: high aleatoric 허용

**측정**:
- MC generation (different temperature/dropout): disagreement = uncertainty
- Semantic uncertainty (Kuhn 2023): clusters in generation distribution → epistemic proxy
- Calibrated LLM: temperature scaling (Ch7-04)

**Mitigation**:
- Retrieval-augmented: epistemic → 외부 지식
- RLHF with uncertainty penalty
- Refuse to answer at high epistemic

**Open problem**: LLM에서 두 uncertainty의 명확 분리 어려움 (weight posterior intractable). Frontier research.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Probabilistic Programming 언어](./02-probabilistic-programming.md) | [📚 README로 돌아가기](../README.md) | [04. OOD Detection과 Calibration ▶](./04-ood-calibration.md) |

</div>
