# 02. VAE의 변종 — β-VAE, CVAE, VQ-VAE

## 🎯 핵심 질문

- **β-VAE**(Higgins et al. 2017)의 $\beta$ 가중치는 ELBO와 어떤 관계인가? 왜 **disentanglement**를 유도하는가?
- **Conditional VAE**(CVAE)는 label $y$를 어떻게 통합하는가?
- **VQ-VAE**(van den Oord 2017)의 **discrete latent** + straight-through estimator는 어떻게 작동하는가?
- 각 변종의 **ELBO 유도**와 학습상의 trade-off는?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

β-VAE는 **interpretable representation learning**의 출발점. Disentangled factor(예: 회전, 크기, 색) 각각을 latent 차원에 분리 → downstream task 유리. CVAE는 **controllable generation**(class-specific digit, style transfer). VQ-VAE는 **discrete token** 기반 생성 — 최근 VQ-VAE는 LLM의 multimodal token 연결(DALL-E, Parti)에서 핵심.

---

## 📐 수학적 선행 조건

- [Ch3-01 VAE](./01-vae-derivation.md)
- [Ch2-06 REINFORCE](../ch2-variational-inference/06-reinforce-control-variate.md): discrete latent
- Straight-through estimator: $\nabla f(\text{round}(x)) \approx \nabla f(x)$
- Vector quantization 기초

---

## 📖 직관적 이해

### β-VAE

KL 항에 $\beta$ 가중:

$$\mathcal{L}_\beta(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta\,\text{KL}(q_\phi(z|x)\|p(z))$$

- $\beta > 1$: KL 더 엄격 → $q(z|x) \approx p(z)$ → 각 latent 차원이 **독립적이고 informative**
- $\beta < 1$: reconstruction 우선 → 복잡한 latent 구조

### CVAE

조건 $y$(class label, style 등)를 encoder·decoder 모두에 주입:
- $q_\phi(z|x, y)$, $p_\theta(x|z, y)$, $p(z|y)$ or $p(z)$
- 학습: supervised pair $(x, y)$
- 생성: 원하는 $y$ 고정 후 $z$ 샘플

### VQ-VAE

Continuous latent 대신 **discrete codebook** $\{e_k\}_{k=1}^K$ 사용:
- Encoder 출력 $z_e(x)$를 가장 가까운 codebook vector $e_k$로 양자화
- Decoder는 $e_k$에서 $x$ 복원
- 학습: straight-through gradient + codebook loss + commitment loss

### 요리 비유

- **β-VAE**: "레시피 다양성을 매우 엄격히 규제" → 독립된 "맛 요소"(단맛, 짠맛)가 latent 축에 분리
- **CVAE**: "한식/양식 라벨에 따라 다른 모드로 요리"
- **VQ-VAE**: "유한한 레시피 목록에서 고름" → 언어 token처럼 이산적

---

## ✏️ 엄밀한 정의

### 정의 2.1 — β-VAE Loss

$$\mathcal{L}_\beta(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta\,\text{KL}(q_\phi(z|x)\|p(z))$$

$\beta = 1$: 표준 VAE. $\beta > 1$: disentanglement.

### 정의 2.2 — CVAE ELBO

조건 $y$ 주어진 데이터 분포:

$$\mathcal{L}(x, y) = \mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - \text{KL}(q_\phi(z|x, y)\|p(z|y))$$

$p(z|y)$ or $p(z)$: 설계 선택. $p(z|y)$를 학습 가능하게 하면 더 유연.

### 정의 2.3 — VQ-VAE Model

- **Encoder**: $z_e(x) \in \mathbb{R}^d$
- **Codebook**: $E = \{e_1, \ldots, e_K\}, e_k \in \mathbb{R}^d$
- **Quantization**: $z_q(x) = e_{k^*}$, $k^* = \arg\min_k\|z_e(x) - e_k\|$
- **Decoder**: $p_\theta(x|z_q)$

Loss:
$$\mathcal{L}_{VQ} = \underbrace{\|x - \hat x\|^2}_{\text{reconstruction}} + \underbrace{\|\text{sg}[z_e] - e\|^2}_{\text{codebook}} + \beta\underbrace{\|z_e - \text{sg}[e]\|^2}_{\text{commitment}}$$

$\text{sg}[\cdot]$ = stop-gradient.

---

## 🔬 정리와 증명

### 정리 2.1 — β-VAE와 Rate-Distortion

**명제** (Alemi et al. 2018): β-VAE는 **rate-distortion**의 Lagrangian과 동치:

$$\min_\phi \mathbb{E}[D(x, \hat x)] \text{ s.t. } I(x; z) \leq R$$

라그랑지안:
$$\mathbb{E}[D] + \beta\,I(x; z)$$

여기서 $I(x; z) \approx \text{KL}(q_\phi(z|x)\|p(z))$ (aggregate variational approximation).

**증명 스케치**: Distortion $D = -\log p(x|z)$, rate $R = \text{KL}(q_\phi(z|x)\|p(z))$. β = Lagrange multiplier. $\square$

**귀결**: β로 "정보 bottleneck"의 capacity 조정. 큰 β → 저용량 latent → disentangled.

### 정리 2.2 — β-VAE는 $\log p(x)$의 ELBO 아님 (when β ≠ 1)

**명제**: $\beta \neq 1$이면 $\mathcal{L}_\beta$는 **진짜 log-evidence의 lower bound가 아니다**.

**증명**: 
$$\log p(x) = \mathcal{L}_1(x) + \text{KL}(q\|p(\cdot|x))$$

$\mathcal{L}_\beta$는 $\mathcal{L}_1$과 KL만큼 다름. 일반적으로 $\beta > 1$이면 $\mathcal{L}_\beta < \mathcal{L}_1 \leq \log p(x)$, **lower bound 관계 유지**. $\beta < 1$이면 $\mathcal{L}_\beta > \mathcal{L}_1$ 가능 — $\log p(x)$의 upper bound도 아님.

실전적으로 $\mathcal{L}_\beta$는 "**다른** objective의 좋은 하한"으로 해석 (정리 2.1). $\square$

### 정리 2.3 — Disentanglement Metric

**명제** (Higgins et al. 2017): Ground-truth factors $f_1, \ldots, f_K$가 있다 할 때, disentanglement는:

$$\text{Score} = \text{accuracy of predicting } f_i \text{ from } z_j\text{'s statistics}$$

β-VAE가 일정 범위의 β에서 **표준 VAE보다 높은 disentanglement score** 보임 (empirically).

**비고**: 이론적 식별 가능성(identifiability)은 비보장 — Locatello et al. 2019은 **un-supervised disentanglement가 inductive bias 없이 불가능**함을 증명.

### 정리 2.4 — CVAE 학습 목표

**명제**: Paired data $(x, y)$에 대해:

$$\log p_\theta(x|y) \geq \mathcal{L}(x, y) = \mathbb{E}_{q_\phi(z|x, y)}[\log p_\theta(x|z, y)] - \text{KL}(q_\phi(z|x, y)\|p(z|y))$$

**증명**: Ch2-01 정리 1.1 with $p(\cdot|y)$ instead of $p(\cdot)$. $\square$

### 정리 2.5 — VQ-VAE Straight-Through Estimator

**명제**: $z_q = e_{k^*}$의 argmin은 **미분 불가능**. Straight-through gradient:

$$\nabla_{z_e}\mathcal{L}_{\text{recon}} \approx \nabla_{z_q}\mathcal{L}_{\text{recon}}$$

즉 decoder의 gradient를 **quantization을 통과해서 encoder까지** 직접 복사.

**증명**: 구현 상 $z_q = z_e + \text{sg}[z_q - z_e]$. Forward pass에선 $z_q$, backward pass에선 $\nabla z_q = \nabla z_e$. Identity gradient. $\square$

### 정리 2.6 — VQ-VAE ELBO Interpretation

**명제**: VQ-VAE는 **categorical latent VAE**에서 **uniform posterior** 가정 + MAP inference로 볼 수 있음 (van den Oord 2017 Appendix).

Posterior $q_\phi(z = k | x) = \delta_{k = k^*(x)}$ (deterministic)가 one-hot. KL(uniform prior $p(z)$, deterministic $q$)은 $\log K$ (상수).

**귀결**: VAE ELBO ≈ reconstruction + constant → VQ-VAE는 **reconstruction만 최적화** + codebook loss + commitment loss로 quantization 정합성 유지.

### 예시 — β-VAE의 KL vs Reconstruction Trade-off

MNIST에서 학습:
- β = 0.5: Reconstruction 좋음, latent 엉킴
- β = 1: 표준 VAE 균형
- β = 4: Reconstruction 약간 나쁨, 각 latent 축이 회전·두께·크기 등 **독립 factor**에 대응

---

## 💻 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────────────────────────────────────
# β-VAE (Ch3-01의 VAE 코드에서 β만 추가)
# ────────────────────────────────────────────────
def beta_vae_loss(recon, x, mu, logvar, beta=4.0):
    BCE = F.binary_cross_entropy(recon, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + beta * KLD

# ────────────────────────────────────────────────
# CVAE — class-conditional generation
# ────────────────────────────────────────────────
class CVAE(nn.Module):
    def __init__(self, input_dim=784, n_classes=10, hidden=400, latent=20):
        super().__init__()
        self.n_classes = n_classes
        self.enc1 = nn.Linear(input_dim + n_classes, hidden)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent + n_classes, hidden)
        self.dec2 = nn.Linear(hidden, input_dim)

    def one_hot(self, y):
        return F.one_hot(y, self.n_classes).float()

    def encode(self, x, y):
        y_oh = self.one_hot(y)
        h = F.relu(self.enc1(torch.cat([x, y_oh], dim=1)))
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z, y):
        y_oh = self.one_hot(y)
        h = F.relu(self.dec1(torch.cat([z, y_oh], dim=1)))
        return torch.sigmoid(self.dec2(h))

    def forward(self, x, y):
        mu, logvar = self.encode(x, y)
        z = mu + torch.randn_like(mu) * torch.exp(0.5*logvar)
        return self.decode(z, y), mu, logvar

# ────────────────────────────────────────────────
# VQ-VAE — discrete latent
# ────────────────────────────────────────────────
class VectorQuantizer(nn.Module):
    def __init__(self, K=512, D=64, beta=0.25):
        super().__init__()
        self.K, self.D, self.beta = K, D, beta
        self.embedding = nn.Embedding(K, D)
        self.embedding.weight.data.uniform_(-1.0/K, 1.0/K)

    def forward(self, z_e):
        # z_e: (B, D) or (B, D, H, W) — flatten if needed
        z_e_flat = z_e.view(-1, self.D)          # (N, D)
        # Distances to all codebook vectors
        dists = (z_e_flat.pow(2).sum(1, keepdim=True)
                 - 2 * z_e_flat @ self.embedding.weight.T
                 + self.embedding.weight.pow(2).sum(1))
        k_star = dists.argmin(dim=1)             # (N,)
        z_q_flat = self.embedding(k_star)        # (N, D)

        # Losses: codebook + commitment
        codebook_loss = F.mse_loss(z_q_flat, z_e_flat.detach())
        commit_loss = F.mse_loss(z_e_flat, z_q_flat.detach())
        vq_loss = codebook_loss + self.beta * commit_loss

        # Straight-through: z_q with identity backward for z_e
        z_q_flat = z_e_flat + (z_q_flat - z_e_flat).detach()

        return z_q_flat.view_as(z_e), vq_loss, k_star
```

---

## 🔗 AI/ML 연결

### Factor VAE, TC-VAE
β-VAE의 개선 — total correlation 항 명시적 추가 (Kim & Mnih 2018).

### DALL-E / Parti
VQ-VAE latent code + autoregressive transformer (LLM)로 이미지 생성. Discrete token이 **language model과 호환**.

### Semi-supervised VAE
CVAE + 부분 라벨 데이터. Semi-supervised classification의 Bayesian 해석.

### Hierarchical VAE (NVAE, Ladder VAE)
여러 latent layer. 각 layer가 서로 다른 추상화 수준.

### Flow-based generative
Normalizing flow (Ch3-03)가 VAE의 Gaussian encoder 제약 완화.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| β 선택 | Hyperparameter tuning 필요 (disentanglement-reconstruction trade) |
| Ground-truth factor 존재 | Unsupervised disentanglement는 ill-posed (Locatello 2019) |
| CVAE: $y$ 깨끗 | Noisy labels에 취약 |
| VQ-VAE: codebook collapse | 일부 vector만 사용, 다른 것 죽음 |
| STE의 biased gradient | 실전 수렴 잘 되지만 이론적 엄밀성 없음 |

---

## 📌 핵심 정리

$$\boxed{\mathcal{L}_\beta = \mathbb{E}_q[\log p(x|z)] - \beta\,\text{KL}(q\|p(z))}$$

세 변종 요약:
- **β-VAE**: $\beta > 1$ → disentanglement (information bottleneck)
- **CVAE**: 조건 $y$를 encoder/decoder 모두에 주입 → controllable generation
- **VQ-VAE**: Discrete codebook + straight-through → token-based generation

---

## 🤔 생각해볼 문제

**문제 1** (기초): β-VAE에서 $\beta \to \infty$의 극한은?

<details>
<summary>해설</summary>

$\beta$가 매우 크면 KL → 0이 최우선 → $q_\phi(z|x) \approx p(z)$.

⇒ Latent가 $x$ 정보를 전혀 담지 못함 (**complete posterior collapse**). Reconstruction은 $p(x|z), z \sim p(z)$ — $x$에 무관한 평균 이미지.

실전: $\beta$는 중간값 (1~10)이 최적. 너무 크면 전체적으로 **무의미한 학습**.

</details>

**문제 2** (심화): CVAE에서 $p(z|y) = \mathcal{N}(\mu_y, I)$로 학습 가능한 class prior를 두면 무엇이 달라지는가?

<details>
<summary>해설</summary>

Class-specific latent cluster가 자연스럽게 형성.

ELBO:
$$\mathbb{E}_q[\log p(x|z, y)] - \text{KL}(q(z|x, y)\|\mathcal{N}(\mu_y, I))$$

각 $y$마다 latent 공간에서 **다른 중심**을 가짐 → generation 시 class-specific region에서 샘플.

장점: Few-shot generation (새 $y$에 적은 data로 $\mu_y$만 조정), interpretable cluster.

단점: $y$ 수가 많으면 $\mu_y$ 학습 많아짐, posterior collapse 여전.

응용: Music generation with "composer style" $y$, text-to-image with caption embedding.

</details>

**문제 3** (AI 연결): DALL-E는 VQ-VAE와 transformer로 이미지 생성. 왜 continuous VAE 대신 VQ-VAE를 썼나?

<details>
<summary>해설</summary>

**Continuous latent**: autoregressive model이 어려움 (각 pixel/latent 의 unbounded continuous output).

**Discrete token (VQ-VAE)**: 
- Transformer의 **next-token prediction** 그대로 적용
- LLM과 같은 architecture (BPE token ↔ codebook idx)
- 실전 inference에서 sampling simple (argmax from softmax over K codes)

단점:
- Codebook 크기 $K$ 제한 → 표현력 cap
- Codebook collapse risk

DALL-E 2 이후엔 Diffusion이 더 인기(연속적 process + CLIP guidance).

이것이 "**왜 VQ-VAE가 multimodal LLM의 backbone**"인지의 핵심. 최근 Whisper, AudioLM, MusicLM 등 audio·video에도 동일한 pattern.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 01. VAE 완전 유도](./01-vae-derivation.md) | [📚 README로 돌아가기](../README.md) | [03. Normalizing Flows ▶](./03-normalizing-flows.md) |

</div>
