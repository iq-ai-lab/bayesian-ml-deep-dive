# 01. VAE 완전 유도 (Kingma & Welling 2013)

## 🎯 핵심 질문

- VAE의 ELBO는 어떻게 **reconstruction + KL**로 분해되는가?
- Gaussian $q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \sigma_\phi^2(x))$과 standard-normal prior $p(z) = \mathcal{N}(0, I)$의 KL은 왜 **해석해**를 갖는가?
- VAE의 **learnable 목적함수**와 **학습 알고리즘 전체**는?
- VAE가 다른 latent variable model (PCA, ICA)과 어떻게 다른가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

VAE는 **generative model의 가장 성공적인 Bayesian 응용** 중 하나. 이미지 생성, 분자 설계, 음악 생성, anomaly detection — 모두 VAE 기반. **Diffusion Model**(Ch7-01)도 VAE의 **hierarchical 확장**으로 볼 수 있어, VAE 이해가 현대 생성모델 이해의 출발점. Bayesian data analysis의 "latent variable" 관점이 실전 딥러닝에 가장 깊이 침투한 예.

---

## 📐 수학적 선행 조건

- [Ch2-01~05](../ch2-variational-inference/01-vi-idea-elbo.md): VI, ELBO, reparameterization
- [Ch1-03 Conjugate priors](../ch1-bayesian-foundation/03-conjugate-priors.md): Gaussian-Gaussian structure
- Neural network basics (autoencoders, softmax)
- [Information Theory Deep Dive](https://github.com/iq-ai-lab/information-theory-deep-dive): KL divergence

---

## 📖 직관적 이해

### VAE의 구조

```
x ──→ Encoder q_φ(z|x) ──→ (μ(x), σ(x)) ──→ z ~ N(μ, σ²)  (reparam)
                                             ↓
                              Decoder p_θ(x|z) ──→ x̂
```

두 NN: **encoder** (inference network)와 **decoder** (generative network).

### Two networks, one loss

- **Encoder** $q_\phi(z|x)$: 관측 $x$를 보고 latent posterior 근사
- **Decoder** $p_\theta(x|z)$: latent $z$에서 관측 복원
- **Loss** (per $x$): $-\text{ELBO}(x) = -\mathbb{E}_q[\log p_\theta(x|z)] + \text{KL}(q_\phi(z|x)\|p(z))$

### 왜 "autoencoder"인가

- 표준 autoencoder: $x \to z \to \hat x$, MSE loss
- VAE: latent를 **확률분포** $q_\phi(z|x)$로 모델링, **KL regularization**으로 prior에 가깝게

KL 덕분에 latent space가 smooth하고 **generative**(prior에서 샘플해도 의미있는 $x$).

### 요리 비유

- 일반 autoencoder: "재료 → 레시피 → 재료" (정확한 복원)
- VAE: "재료 → **레시피의 분포** → 재료". 레시피에 variability가 허용되어 **새 재료** 생성 가능.

---

## ✏️ 엄밀한 정의

### 정의 1.1 — VAE Model

**Generative model**:
$$p_\theta(x, z) = p_\theta(x|z)p(z)$$

- Prior: $p(z) = \mathcal{N}(0, I)$ (보통 standard normal)
- Decoder (conditional likelihood): $p_\theta(x|z)$ = neural network with params $\theta$
  - 이미지: $\mathcal{N}(\text{dec}_\theta(z), \sigma^2 I)$ or $\prod\text{Bernoulli}(\pi_\theta(z))$

### 정의 1.2 — Variational Inference Model

$$q_\phi(z|x) = \mathcal{N}(\mu_\phi(x), \text{diag}(\sigma_\phi^2(x)))$$

$\mu_\phi, \log\sigma_\phi^2$ = neural network output(encoder).

### 정의 1.3 — VAE ELBO (per datum)

$$\mathcal{L}(x; \theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x)\|p(z))$$

### 정의 1.4 — VAE Training Objective

$$\max_{\theta, \phi}\sum_{i=1}^N \mathcal{L}(x_i; \theta, \phi)$$

SGD + reparameterization + mini-batch.

---

## 🔬 정리와 증명

### 정리 1.1 — VAE ELBO 유도

**명제**: $\log p_\theta(x) \geq \mathcal{L}(x; \theta, \phi)$:

$$\log p_\theta(x) \geq \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x)\|p(z))$$

**증명**: Ch2-01 정리 1.1 + 분해 (2). $\log p_\theta(x) = \mathcal{L}(x) + \text{KL}(q\|p(\cdot|x, \theta)) \geq \mathcal{L}(x)$. $\square$

### 정리 1.2 — Gaussian KL Closed Form

**명제**: $q_\phi(z|x) = \mathcal{N}(\mu, \text{diag}(\sigma^2)), p(z) = \mathcal{N}(0, I)$에 대해 (각 $j$):

$$\text{KL}(q_\phi(z|x)\|p(z)) = \frac{1}{2}\sum_{j=1}^d (\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$$

**증명**:

일반 $\mathcal{N}(\mu, \Sigma)$와 $\mathcal{N}(0, I)$의 KL:
$$\text{KL} = \frac{1}{2}[\text{tr}(\Sigma) + \mu^T\mu - k - \log|\Sigma|]$$

$\Sigma = \text{diag}(\sigma^2)$:
$$= \frac{1}{2}\left[\sum_j \sigma_j^2 + \sum_j \mu_j^2 - d - \sum_j\log\sigma_j^2\right]$$

$$= \frac{1}{2}\sum_j (\sigma_j^2 + \mu_j^2 - \log\sigma_j^2 - 1)$$

$\square$

> **핵심**: KL이 **해석해** → MC 추정 불필요 → **저분산** 학습. 이것이 VAE가 학습 가능한 결정적 이유.

### 정리 1.3 — Reconstruction Term의 MC 추정

**명제**: Reconstruction은 $q_\phi(z|x)$에서 MC 샘플:

$$\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] \approx \frac{1}{L}\sum_{l=1}^L \log p_\theta(x|z^{(l)}), \quad z^{(l)} = \mu_\phi + \sigma_\phi\odot\epsilon^{(l)}, \epsilon^{(l)}\sim\mathcal{N}(0, I)$$

실전: **$L = 1$**으로 충분 (mini-batch가 이미 averaging). Reparameterization trick 덕분에 **gradient가 encoder로 흐름** (Ch2-05).

### 정리 1.4 — Decoder의 Loss 형태

Gaussian decoder $p_\theta(x|z) = \mathcal{N}(\text{dec}_\theta(z), I)$:
$$-\log p_\theta(x|z) = \frac{1}{2}\|x - \text{dec}_\theta(z)\|^2 + \text{const}$$
⇒ **MSE reconstruction loss**.

Bernoulli decoder $p_\theta(x|z) = \prod_j \text{Ber}(\pi_{\theta,j}(z))$:
$$-\log p_\theta(x|z) = -\sum_j[x_j\log\pi_j + (1-x_j)\log(1-\pi_j)]$$
⇒ **Binary cross-entropy**.

### 정리 1.5 — VAE vs PCA

**명제**: Linear VAE with $\text{dec}_\theta(z) = Wz$ (Gaussian noise, isotropic prior)는 **Probabilistic PCA** (Tipping & Bishop 1999)와 동치.

**증명 개요**: Linear-Gaussian 모델의 conjugate 구조 → exact posterior $p(z|x)$도 Gaussian. VAE encoder가 이 Gaussian을 정확히 회복할 수 있음. 해석해로 $W$가 data covariance의 principal components. $\square$

**귀결**: VAE = **nonlinear PCA with stochastic encoder**. 표현력이 PCA 훨씬 뛰어넘음.

### 예시 — MNIST VAE

Standard setup:
- Encoder: MLP $784 \to 400 \to (\mu_d, \log\sigma_d^2)$ with $d = 20$
- Decoder: MLP $20 \to 400 \to 784$ (Bernoulli output for binary MNIST)
- Loss: BCE + KL (정리 1.4 + 1.2)
- Optimizer: Adam, learning rate $10^{-3}$

---

## 💻 PyTorch 구현 (실전 VAE)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.enc1 = nn.Linear(input_dim, hidden_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec1 = nn.Linear(latent_dim, hidden_dim)
        self.dec2 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon, x, mu, logvar):
    # Reconstruction — BCE summed over pixels
    BCE = F.binary_cross_entropy(recon, x, reduction='sum')
    # KL term — closed form (정리 1.2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD

def train_vae(epochs=10, batch_size=128):
    transform = transforms.Compose([transforms.ToTensor(), 
                                     transforms.Lambda(lambda x: x.view(-1))])
    loader = DataLoader(datasets.MNIST('./data', train=True, download=True, 
                                        transform=transform),
                        batch_size=batch_size, shuffle=True)
    
    model = VAE()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for epoch in range(epochs):
        total_loss = 0
        for x, _ in loader:
            optimizer.zero_grad()
            recon, mu, logvar = model(x)
            loss = vae_loss(recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}: avg loss = {total_loss/len(loader.dataset):.2f}")
    return model

# 학습 후: prior에서 샘플 → 이미지 생성
# model.eval(); z = torch.randn(16, 20); x_gen = model.decode(z).view(16, 28, 28)
```

**학습 진단**:
- **Reconstruction loss**과 **KL loss**를 **따로 추적**
- KL → 0이면 posterior collapse 의심
- Reconstruction 감소 → latent가 유용한 정보 담음

---

## 🔗 AI/ML 연결

### β-VAE (Ch3-02)
$\beta > 1$로 disentanglement 유도.

### Conditional VAE
$q_\phi(z|x, y), p_\theta(x|z, y)$ — labeled generation. CVAE의 직접 확장.

### Hierarchical VAE
여러 latent layer → 더 깊은 representation. Ladder VAE, NVAE.

### Diffusion Model
DDPM = hierarchical VAE with $T$ stages + Gaussian transitions. VAE ELBO의 T-step 확장 (Ch7-01).

### Autoencoding for Semi-supervised Learning
Kingma et al. 2014 — VAE representation + labeled fine-tuning.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $p(z) = \mathcal{N}(0, I)$ | 단순 prior → 표현력 제한 (VampPrior 등 대안) |
| Mean-field Gaussian $q$ | Posterior correlation 무시 |
| Decoder가 Gaussian/Bernoulli | 복잡한 likelihood 구현 복잡 |
| **Posterior collapse** 위험 | 강력한 decoder일수록 더 흔함 |
| **Blurry samples** | MSE-like loss의 결과 |

**실전 팁**: 
- **KL annealing**: 초기엔 KL 계수 0부터 1까지 증가 (collapse 방지)
- **Free bits**: KL per-dimension에 minimum threshold

---

## 📌 핵심 정리

$$\boxed{\mathcal{L}(x) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x)\|p(z))}$$

$$\boxed{\text{KL}(q\|p) = \frac{1}{2}\sum_j(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)}$$

핵심:
- **Reconstruction + KL** 형태
- Gaussian prior + Gaussian posterior → **KL 해석해**
- Reparameterization으로 **gradient 흐름 유지**
- **Latent space**가 smooth & generative

---

## 🤔 생각해볼 문제

**문제 1** (기초): VAE의 latent dimension $d$가 너무 작으면 무슨 문제가 생기나? 너무 크면?

<details>
<summary>해설</summary>

**너무 작음** ($d$ 작음): 표현력 부족 → reconstruction 품질 나쁨, blurry. MNIST의 경우 $d < 5$이면 숫자별 구분 어려움.

**너무 큼** ($d$ 큼): **Posterior collapse**. 과잉 차원이 prior로 "붕괴". Decoder가 무시.

Sweet spot: MNIST $d \approx 10-20$, CelebA $d \approx 100-500$. Ablation study로 찾는 것이 표준.

Hierarchical VAE는 $d$를 효과적으로 확장 (여러 layer).

</details>

**문제 2** (심화): "Reparameterization trick 없이" VAE를 학습하려면? 왜 그게 실패하는가?

<details>
<summary>해설</summary>

없이 → REINFORCE (Ch2-06): $\nabla_\phi \mathbb{E}_q[\log p_\theta(x|z)] = \mathbb{E}_q[\log p_\theta(x|z) \cdot \nabla_\phi\log q_\phi(z|x)]$.

**문제**: $\log p_\theta(x|z)$ 값이 매우 크거나 작은 경우 (NN 출력), gradient의 분산이 거대 → SGD 수렴 안 함.

Reparam은 shared randomness로 variance를 획기적으로 낮춤 (Ch2-05 정리 5.3). 이것 없이 VAE는 사실상 **학습 불가**.

이것이 2013년 VAE 원전 논문의 "기여의 기술적 핵심" — ELBO 자체는 잘 알려진 것이었지만, 신경망 encoder에 gradient를 **흘릴 방법**이 reparam이었음.

</details>

**문제 3** (AI 연결): **Diffusion Model**의 ELBO가 VAE ELBO를 $T$ step 반복한 것이라 한다. 구체적으로 어떻게?

<details>
<summary>해설</summary>

DDPM:
- Forward (encoder 없음): $q(x_t|x_{t-1}) = \mathcal{N}(\sqrt{1-\beta_t}x_{t-1}, \beta_t I)$ — 고정 Markov
- Reverse (decoder): $p_\theta(x_{t-1}|x_t)$ — learnable Markov
- ELBO (per $x_0$):
$$\log p_\theta(x_0) \geq \mathbb{E}_q\left[\log\frac{p_\theta(x_0, \ldots, x_T)}{q(x_1, \ldots, x_T|x_0)}\right]$$

전개하면:
$$= -\text{KL}(q(x_T|x_0)\|p(x_T)) - \sum_{t=1}^T \mathbb{E}_q[\text{KL}(q(x_{t-1}|x_t, x_0)\|p_\theta(x_{t-1}|x_t))] + \mathbb{E}_q[\log p_\theta(x_0|x_1)]$$

첫 항 = prior match, 둘째 = **T개의 "VAE ELBO"와 유사한** denoising KL, 셋째 = 마지막 reconstruction.

즉 **hierarchical VAE with $T$ levels**. Ch7-01에서 이 유도를 DSM과 연결.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ Ch2-06 REINFORCE와 Control Variate](../ch2-variational-inference/06-reinforce-control-variate.md) | [📚 README로 돌아가기](../README.md) | [02. VAE의 변종 — β-VAE, CVAE, VQ-VAE ▶](./02-vae-variants.md) |

</div>
