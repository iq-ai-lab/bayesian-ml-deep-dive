# 04. OOD Detection과 Calibration

## 🎯 핵심 질문

- Bayesian predictive가 OOD에서 **높은 epistemic uncertainty**를 보이는 이유는?
- **ECE** $= \sum_m \frac{|B_m|}{n}|\text{acc}(B_m) - \text{conf}(B_m)|$의 정의와 측정?
- **Temperature scaling** (Guo et al. 2017) vs deep ensembles의 calibration 비교?
- Bernstein-von Mises 기반으로 왜 Bayesian이 **자동 calibrated**인가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**"모델의 confidence가 믿을 만한가?"** — reliability의 근본. Calibration 부족은 의료·법·금융에서 치명적. OOD detection은 자율주행에서 **예기치 않은 상황 인지**의 핵심. Bayesian이 이 두 문제에 대한 **원리적 해결**: posterior uncertainty가 자동으로 calibrated (BvM).

---

## 📐 수학적 선행 조건

- [Ch7-03](./03-epistemic-aleatoric.md): Epistemic/Aleatoric
- [Ch5 전체](../ch5-bayesian-nn/01-bnn-formulation.md): BNN
- [Ch1-05 BvM](../ch1-bayesian-foundation/05-bernstein-von-mises.md): asymptotic calibration

---

## 📖 직관적 이해

### Calibration

"Model이 95% 확신하는 예측이 실제로 95% 맞는가?"

- **Perfectly calibrated**: $\text{confidence} = \text{accuracy}$
- **Over-confident** (modern NN): confidence > accuracy
- **Under-confident**: confidence < accuracy

### ECE (Expected Calibration Error)

Bin-based metric:
$$\text{ECE} = \sum_m \frac{|B_m|}{n}|\text{acc}(B_m) - \text{conf}(B_m)|$$

- 예측을 confidence로 $M$ bins 분할
- 각 bin에서 accuracy vs average confidence 차이
- Weighted sum.

### OOD Detection

Training distribution 밖 입력 식별. Bayesian의 자연 방법:
- **Epistemic uncertainty** → OOD 높음
- **Softmax entropy** → 분류 확신도 낮음
- **Density-based** (VAE ELBO, likelihood)

### 요리 비유

- Calibration: "70% 확신 → 70% 맞음" — honest
- OOD: "이 재료 처음 봐 → 확신 낮음"

---

## ✏️ 엄밀한 정의

### 정의 4.1 — Calibration

$$\forall c \in [0, 1]: P(\hat y = y | \hat p = c) = c$$

$\hat p$ = predicted probability, $\hat y$ = argmax prediction.

### 정의 4.2 — Expected Calibration Error

$M$ bins $B_1, \ldots, B_M$ (confidence intervals):

$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{n}|\text{acc}(B_m) - \text{conf}(B_m)|$$

- $\text{acc}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m}\mathbf{1}[\hat y_i = y_i]$
- $\text{conf}(B_m) = \frac{1}{|B_m|}\sum_{i \in B_m}\hat p_i$

### 정의 4.3 — Maximum Calibration Error

$$\text{MCE} = \max_m|\text{acc}(B_m) - \text{conf}(B_m)|$$

최악 bin의 deviation.

### 정의 4.4 — Temperature Scaling

Logits $z$에 scalar $T > 0$:
$$\hat p = \text{softmax}(z/T)$$

$T$는 validation set에서 NLL 최소화로 학습.

- $T > 1$: soften (over-confident 모델 수정)
- $T < 1$: sharpen

### 정의 4.5 — OOD Score

- **Max softmax**: $\max_c p(y=c|x)$ — 낮을수록 OOD
- **Predictive entropy**: $H[p(y|x)]$ — 클수록 OOD
- **Epistemic**: (정리 3.5) — 클수록 OOD
- **Density**: $\log p(x)$ (VAE, flow) — 낮을수록 OOD

---

## 🔬 정리와 증명

### 정리 4.1 — Bayesian이 Asymptotically Calibrated

**명제**: BvM 조건 하 (Ch1-05), $N \to \infty$에서 Bayesian predictive interval의 **frequentist coverage**가 target과 일치.

**증명**: BvM → posterior ≈ $\mathcal{N}(\hat\theta_{MLE}, F^{-1}/N)$. Bayesian $(1-\alpha)$-credible interval = frequentist $(1-\alpha)$-CI asymptotically. Coverage가 $1-\alpha$ → calibrated. $\square$

**귀결**: Bayesian은 대규모 data에서 **자동 calibrated** — hand-tuning 불필요.

### 정리 4.2 — Modern NN이 Over-Confident인 이유

**Guo et al. 2017 empirical**: 
- Modern NN (ResNet, Transformer)의 $\max p$ 평균이 accuracy보다 $10\sim 20\%$ 높음
- Capacity 증가, batch norm, weight decay가 miscalibration 악화

**이론적 설명** (Laplace 관점):
- Over-parameterized → weight posterior가 좁아짐
- Single-mode MAP이 실제 불확실성 과소 반영
- Full Bayesian과 차이 (정리 4.1 위반)

### 정리 4.3 — Temperature Scaling의 불변성

**명제**: Temperature scaling은 **argmax를 바꾸지 않음**:

$$\arg\max_c \text{softmax}(z/T)_c = \arg\max_c z_c$$

**증명**: $\text{softmax}(z/T) = e^{z/T}/\sum e^{z_c/T}$. $T > 0$ 단조 변환 → argmax 불변. $\square$

**귀결**: Accuracy 변하지 않음, confidence만 조정 → Calibration 개선 without accuracy loss.

### 정리 4.4 — Deep Ensembles의 Calibration

**명제** (Lakshminarayanan et al. 2017): $M$개 독립 학습된 NN의 predictive average가 single NN보다 **일반적으로 더 calibrated**.

**근거**:
- Ensemble diversity → multi-mode posterior 근사
- 각 model의 over-confidence가 average에서 완화
- Random init + random data order로 diverse solutions

**수식** (informal): 
$$\bar p(y|x) = \frac{1}{M}\sum_i p_{\theta_i}(y|x)$$

ECE($\bar p$) ≤ ECE(single $p$) empirically.

### 정리 4.5 — OOD와 Epistemic의 관계

**명제**: 잘 학습된 BNN에서 OOD input $x^*$:

$$\sigma_{\text{epi}}^2(x^*) \gg \sigma_{\text{epi}}^2(\text{training } x)$$

**근거**:
- Training 영역은 observed → posterior 좁음 → low epistemic
- OOD 영역은 never seen → posterior 넓음 → high epistemic

**실전**: OOD threshold = training에서 epistemic 분포의 95% quantile + safety margin.

---

## 💻 ECE 측정 및 Temperature Scaling

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ────────────────────────────────────────────────
# Compute ECE
# ────────────────────────────────────────────────
def ece(confidences, predictions, labels, n_bins=15):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers, bin_uppers = bin_boundaries[:-1], bin_boundaries[1:]
    ece_value = 0
    bin_data = []
    for low, high in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > low) & (confidences <= high)
        n_in = in_bin.sum()
        if n_in > 0:
            acc = (predictions[in_bin] == labels[in_bin]).mean()
            conf = confidences[in_bin].mean()
            ece_value += (n_in / len(labels)) * abs(acc - conf)
            bin_data.append((low + high)/2, acc, conf, n_in)
    return ece_value, bin_data

# ────────────────────────────────────────────────
# Temperature Scaling
# ────────────────────────────────────────────────
class TemperatureScaling:
    def __init__(self):
        self.T = torch.nn.Parameter(torch.ones(1) * 1.5)
    
    def fit(self, logits_val, labels_val, n_iter=100):
        opt = torch.optim.LBFGS([self.T], lr=0.01, max_iter=n_iter)
        criterion = nn.CrossEntropyLoss()
        def closure():
            opt.zero_grad()
            loss = criterion(logits_val / self.T, labels_val)
            loss.backward()
            return loss
        opt.step(closure)
        return self.T.item()
    
    def scale(self, logits):
        return logits / self.T

# ────────────────────────────────────────────────
# Reliability diagram
# ────────────────────────────────────────────────
def plot_reliability(confidences, predictions, labels, n_bins=15, title=''):
    _, bin_data = ece(confidences, predictions, labels, n_bins)
    centers, accs, confs, counts = zip(*bin_data) if bin_data else ([], [], [], [])
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='perfect')
    ax.bar(centers, accs, width=1/n_bins*0.9, alpha=0.5, edgecolor='k', label='observed')
    ax.plot(centers, confs, 'ro-', label='confidence')
    ax.set_xlabel('Confidence'); ax.set_ylabel('Accuracy')
    ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
    return fig

# ────────────────────────────────────────────────
# Example: synthetic over-confident model
# ────────────────────────────────────────────────
rng = np.random.default_rng(0)
N = 1000
K = 5
# True accuracies by confidence bin
true_probs = rng.uniform(0.1, 0.95, N)  # true correctness probabilities
labels = rng.binomial(1, true_probs)      # actually correct or not (simplified binary)
# Over-confident logits (predict confidence higher than true)
confidences_over = np.minimum(true_probs + 0.15, 0.99)
predictions = (rng.random(N) < confidences_over).astype(int)  # simulate prediction

# Compute ECE
ece_over, _ = ece(confidences_over, predictions, labels)
print(f"Over-confident ECE: {ece_over:.4f}")
```

---

## 🔗 AI/ML 연결

### Medical AI
FDA approval에 calibration evidence 필요. ECE, reliability diagram 제출.

### Autonomous Driving
OOD detection으로 fallback behavior — Bayesian 또는 ensemble의 필요.

### LLM Confidence
Self-consistency (Wang et al. 2023): multiple generations → agreement = calibrated confidence.

### Conformal Prediction
Distribution-free calibration guarantee. Bayesian과 orthogonal technique — 결합 가능.

### Active Learning
OOD-like 점(high epistemic)을 acquisition → labeled data로 model 개선.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| Validation set available | Distribution shift 시 temperature scaling 부정확 |
| i.i.d. test | OOD shift에선 ECE 의미 제한 |
| Single uncertainty source | Epistemic + aleatoric 구분 필요 |
| Discrete bins | Continuous version (ACE) 제안 |

**실전 체크리스트**:
1. ECE 측정 (10~20 bins)
2. Reliability diagram 시각화
3. Temperature scaling 적용 (validation set)
4. ECE 재측정
5. OOD test: OOD score 분포를 in vs out으로 비교

---

## 📌 핵심 정리

$$\boxed{\text{ECE} = \sum_m \frac{|B_m|}{n}|\text{acc}(B_m) - \text{conf}(B_m)|}$$

핵심:
- Modern NN은 **over-confident** (Guo 2017)
- **Temperature scaling**: simple post-hoc fix (argmax 불변)
- **Bayesian**: BvM 하에서 자동 calibrated (정리 4.1)
- **Deep ensembles**: practical calibration improvement
- **OOD**: Epistemic uncertainty가 naturally-higher

---

## 🤔 생각해볼 문제

**문제 1** (기초): ECE = 0.05와 ECE = 0.15 중 어느 것이 더 좋은 모델?

<details>
<summary>해설</summary>

**ECE 0.05** (5% 평균 miscalibration) > **0.15** (15%). 낮을수록 좋음.

보통:
- ECE < 0.01: 잘 calibrated
- ECE 0.01~0.05: 괜찮음
- ECE > 0.1: significant miscalibration

단, **accuracy는 별개**: ECE 낮아도 accuracy 낮을 수 있음. Confidence = accuracy는 "**honest**" 개념, 두 축이 따로.

**Temperature scaling 후**: 주로 ECE 0.01 근처로 감소. Deep ensembles는 추가 개선.

</details>

**문제 2** (심화): Temperature scaling이 OOD에서 작동 안 하는 이유?

<details>
<summary>해설</summary>

Temperature $T$는 validation set(in-distribution)에서 학습. OOD는 distribution이 달라:
- In-dist confidence-accuracy relation과 다른 관계
- $T$가 OOD에선 wrong miscalibration 방향 수정

**예**: In-dist에선 over-confident ($T^* > 1$) → temperature scaling이 soften. OOD에선 아무거나 prediction → temperature가 wrong.

**해결**:
- **Domain-specific T**: domain별 temperature (if label info)
- **Bayesian**: 내재적 uncertainty → OOD 자동 handle (정리 4.5)
- **Ensemble**: diversity → OOD에서 disagreement

OOD detection은 **temperature scaling과 독립** — 다른 tool 필요.

</details>

**문제 3** (AI 연결): LLM의 hallucination detection에 이 framework를?

<details>
<summary>해설</summary>

**LLM 특수성**:
- Token-level prediction: each token has softmax prob
- Autoregressive: sequence-level confidence 어려움
- Self-consistency (Wang 2023): 여러 sample의 agreement = calibrated confidence

**Approaches**:
1. **Token-level ECE**: per-token confidence calibration
2. **Self-consistency**: $T$ samples → vote → agreement ratio = confidence
3. **Semantic uncertainty** (Kuhn 2023): embed generations → cluster → cluster count = epistemic
4. **Verification**: 질문-답변 consistency check

**Hallucination = low calibration + confident output**:
- Token calibration이 OK여도 sentence-level factuality는 별도
- "Factuality calibration"이 새로운 연구 방향

**Best practice (2024)**:
- RAG (retrieval-augmented) → reduce epistemic
- RLHF with uncertainty penalty
- Refuse to answer when self-consistency low

Bayesian 이해가 이 방향의 기초.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 03. Bayesian Deep Learning의 불확실성 분해](./03-epistemic-aleatoric.md) | [📚 README로 돌아가기](../README.md) | 🎉 **완주 축하합니다!** |

</div>
