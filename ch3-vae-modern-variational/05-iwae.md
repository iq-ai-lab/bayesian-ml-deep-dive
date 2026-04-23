# 05. Importance-weighted VAE (IWAE)

## 🎯 핵심 질문

- **IWAE bound** $\mathcal{L}_K = \mathbb{E}\left[\log\frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q(z_k|x)}\right]$는 왜 **tighter** bound인가?
- $\mathcal{L}_K \geq \mathcal{L}_{K-1} \geq \cdots \geq \mathcal{L}_1 = \text{ELBO}$의 **monotonicity**는 어떻게 증명되는가?
- $K \to \infty$에서 $\mathcal{L}_K \to \log p(x)$로의 수렴은?
- IWAE가 실전에서 VAE보다 **tighter inference**를 주지만 왜 **항상 더 나은 학습**은 아닌가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

VAE의 ELBO가 looose (특히 posterior가 Gaussian에서 벗어날 때). IWAE (Burda, Grosse, Salakhutdinov 2016)는 **single forward pass에서 $K$개 샘플**로 더 정확한 bound. 구현 간단, VAE 코드에 몇 줄 추가. 이론적으로 **log-likelihood 정확 계산**의 lower-cost 근사. 최근 Diffusion model의 VLB 개선에도 비슷한 아이디어 적용.

---

## 📐 수학적 선행 조건

- [Ch3-01 VAE](./01-vae-derivation.md)
- Importance sampling 기초
- Jensen 부등식, Monte Carlo 추정
- Law of large numbers

---

## 📖 직관적 이해

### IWAE의 아이디어

표준 ELBO:
$$\mathcal{L}_1 = \mathbb{E}_{q(z|x)}\left[\log\frac{p(x, z)}{q(z|x)}\right]$$

IWAE (K-sample):
$$\mathcal{L}_K = \mathbb{E}_{z_1, \ldots, z_K \sim q}\left[\log\frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q(z_k|x)}\right]$$

**log 밖으로 합 + 안쪽 평균** 구조. 이것이 Jensen 부등식의 "gap을 줄임".

### 왜 Tighter?

$$\log p(x) = \log \mathbb{E}_q[p(x, z)/q(z|x)]$$

- VAE ($K=1$): $\log p(x) \geq \mathbb{E}_q[\log p(x,z)/q]$ — 큰 gap
- IWAE ($K$ big): $\log p(x) \approx \log \frac{1}{K}\sum_k p(x,z_k)/q(z_k|x)$ — MC로 $\log p(x)$에 직접 근사

### "Importance weighting" 관점

Weights: $w_k = p(x, z_k)/q(z_k|x)$. IWAE bound:
$$\mathcal{L}_K = \mathbb{E}[\log\bar w], \quad \bar w = \frac{1}{K}\sum w_k$$

샘플 평균의 log의 기댓값. Strong Law of Large Numbers: $\bar w \to p(x)$ a.s., 따라서 $\log\bar w \to \log p(x)$.

### 요리 비유

VAE = "요리 1개 만들고 평점 계산"
IWAE = "요리 K개 만들고 **최고점수 가중평균**" — 운 나쁜 샘플의 영향 감소, true distribution에 근사

---

## ✏️ 엄밀한 정의

### 정의 5.1 — IWAE Bound

$$\mathcal{L}_K(x) := \mathbb{E}_{z_1, \ldots, z_K \stackrel{iid}{\sim} q(z|x)}\left[\log\frac{1}{K}\sum_{k=1}^K \frac{p(x, z_k)}{q(z_k|x)}\right]$$

$K = 1$: 표준 ELBO ($\mathcal{L}_1$).

### 정의 5.2 — Importance Weights

$$w_k := \frac{p(x, z_k)}{q(z_k|x)} = \frac{p(x|z_k)p(z_k)}{q(z_k|x)}$$

$\mathbb{E}_q[w_k] = p(x)$ (unbiased IS estimator).

### 정의 5.3 — Effective Sample Size

$$\text{ESS}_K = \frac{(\sum_k w_k)^2}{\sum_k w_k^2}$$

IS estimator의 "유효 샘플 수". $\text{ESS} \to K$이면 $q$가 좋음, $\ll K$면 posterior에서 멀음.

---

## 🔬 정리와 증명

### 정리 5.1 — IWAE는 Lower Bound

**명제**: $\forall K \geq 1$:
$$\mathcal{L}_K(x) \leq \log p(x)$$

**증명**:

Jensen 부등식 (concave $\log$):
$$\mathcal{L}_K = \mathbb{E}\left[\log\frac{1}{K}\sum w_k\right] \leq \log\mathbb{E}\left[\frac{1}{K}\sum w_k\right] = \log\mathbb{E}[w_1] = \log p(x)$$

(마지막: $w_k$가 iid에서 $\mathbb{E}[w] = p(x)$, 평균도 동일)

$\square$

### 정리 5.2 — IWAE Monotonicity

**명제**: $\mathcal{L}_K$는 $K$에 대해 **단조 증가**:
$$\mathcal{L}_K \geq \mathcal{L}_{K-1} \geq \cdots \geq \mathcal{L}_1$$

**증명** (Burda et al. 2016, Theorem 1):

$K$-샘플 평균 $\bar w_K = \frac{1}{K}\sum_{k=1}^K w_k$. 특정 $k$를 제외한 $(K-1)$-샘플 평균 $\bar w_{K-1}^{(-k)}$.

**관찰**: $\bar w_K = \frac{1}{K}\sum_k \bar w_{K-1}^{(-k)}$... (정확한 전개: for distinct indices).

실제로 $K$-평균은 $(K-1)$-평균들의 평균:
$$\bar w_K = \mathbb{E}_{k \text{ uniform}}[\bar w_{K-1}^{(-k)}]$$

Jensen (concave $\log$):
$$\log\bar w_K = \log\mathbb{E}_k[\bar w_{K-1}^{(-k)}] \geq \mathbb{E}_k[\log\bar w_{K-1}^{(-k)}]$$

양변에 $\mathbb{E}_{z_1, \ldots, z_K}$ 취하면:
$$\mathcal{L}_K \geq \mathbb{E}_{z_1,\ldots,z_K}[\mathbb{E}_k[\log\bar w_{K-1}^{(-k)}]] = \mathcal{L}_{K-1}$$

(마지막 등식: 각 $k$를 제외한 $K-1$ 샘플은 여전히 iid $q$에서.)

$\square$

### 정리 5.3 — $K \to \infty$ 수렴

**명제**: $w_k$가 uniformly integrable이면:
$$\mathcal{L}_K \to \log p(x) \quad (K \to \infty)$$

**증명**:

$\bar w_K \to \mathbb{E}[w] = p(x)$ a.s. (strong LLN).

$\log$ 연속 + uniform integrability → $\mathbb{E}[\log\bar w_K] \to \log p(x)$. $\square$

**수렴율**: ELBO gap은 $O(1/K)$로 수렴 (Maddison et al. 2017).

### 정리 5.4 — IWAE Gradient의 Importance-Weighted Form

**명제**: IWAE의 gradient (reparameterization trick 후):

$$\nabla_\phi \mathcal{L}_K = \mathbb{E}_{\epsilon}\left[\sum_{k=1}^K \tilde w_k \nabla_\phi \log w_k(z_k(\phi, \epsilon))\right]$$

where $\tilde w_k = w_k / \sum_j w_j$ (normalized weights).

**증명 스케치**: $\nabla \log\frac{1}{K}\sum w_k = \sum_k \tilde w_k \nabla\log w_k$ via $\nabla\log = \nabla w/w$. 그리고 reparam으로 $z_k(\phi, \epsilon_k)$. $\square$

**귀결**: $\tilde w_k$가 weight ⇒ high-weight sample이 gradient를 dominate. 이 구조가 **encoder training signal의 감소**로 이어질 수 있음 (Rainforth 2018).

### 정리 5.5 — IWAE가 Encoder 학습을 느리게 할 수 있음 (Rainforth et al. 2018)

**명제**: $K$가 매우 크면 $\nabla_\phi\mathcal{L}_K$의 **SNR**(signal-to-noise ratio)가 **$O(1/\sqrt K)$로 감소**.

**증명 스케치**: IWAE gradient의 variance가 inherent 증가, 특히 encoder $\phi$에 대해. $\square$

**실전 교훈**: 
- Decoder($\theta$)는 $K$가 커지면 개선
- Encoder($\phi$)는 $K$가 너무 크면 오히려 악화
- 실전 $K \in \{5, 10, 50\}$가 balanced

---

## 💻 PyTorch 구현

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class IWAE(nn.Module):
    def __init__(self, input_dim=784, hidden=400, latent=20, K=10):
        super().__init__()
        self.K = K
        self.enc1 = nn.Linear(input_dim, hidden)
        self.enc_mu = nn.Linear(hidden, latent)
        self.enc_logvar = nn.Linear(hidden, latent)
        self.dec1 = nn.Linear(latent, hidden)
        self.dec2 = nn.Linear(hidden, input_dim)

    def encode(self, x):
        h = F.relu(self.enc1(x))
        return self.enc_mu(h), self.enc_logvar(h)

    def decode(self, z):
        h = F.relu(self.dec1(z))
        return torch.sigmoid(self.dec2(h))

    def log_normal(self, z, mu, logvar):
        return (-0.5*np.log(2*np.pi) - 0.5*logvar 
                - 0.5*(z-mu).pow(2)/logvar.exp()).sum(dim=-1)

    def elbo_K(self, x):
        B = x.shape[0]
        mu, logvar = self.encode(x)
        # Expand for K samples
        mu_k = mu.unsqueeze(1).expand(-1, self.K, -1)       # (B, K, d)
        logvar_k = logvar.unsqueeze(1).expand(-1, self.K, -1)
        eps = torch.randn_like(mu_k)
        z = mu_k + eps * torch.exp(0.5*logvar_k)            # (B, K, d)
        
        # log p(x | z_k): Bernoulli
        z_flat = z.view(B*self.K, -1)
        recon = self.decode(z_flat).view(B, self.K, -1)
        log_px_given_z = -F.binary_cross_entropy(recon, 
                            x.unsqueeze(1).expand(-1, self.K, -1), 
                            reduction='none').sum(dim=-1)    # (B, K)
        
        # log p(z_k) ~ standard normal
        log_pz = -0.5*z.pow(2).sum(dim=-1) - 0.5*z.shape[-1]*np.log(2*np.pi)
        
        # log q(z_k | x)
        log_qz = self.log_normal(z, mu_k, logvar_k)
        
        # log w_k = log p(x, z) - log q(z | x)
        log_w = log_px_given_z + log_pz - log_qz             # (B, K)
        
        # IWAE bound: E[log (1/K) sum w_k] = E[logsumexp(log_w) - log K]
        bound = (torch.logsumexp(log_w, dim=1) - np.log(self.K)).mean()
        return bound

# 학습 loop: loss = -model.elbo_K(x)
# K=1: 표준 VAE, K=5~50: IWAE
```

---

## 🔗 AI/ML 연결

### Variational Importance Sampling
PPL에서 posterior 추정의 고급 방법.

### Diffusion Model VLB Improvement
Nichol & Dhariwal (2021): VLB tight bound, IWAE와 비슷한 multi-sample 기법.

### Expectation Maximization (hard EM vs soft)
Hard EM은 best $z$만, soft EM은 posterior avg — IWAE는 그 중간.

### Sequential Monte Carlo
IWAE는 1-step SMC로 볼 수 있음. Filtering VAE 등 시계열 확장.

### GANs와의 비교
GAN은 likelihood 계산 없이 학습 — IWAE는 tighter likelihood lower bound. 서로 보완.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $q$에서 $K$ 샘플 가능 | Memory $O(K \cdot d)$ 증가 |
| Importance weight 안정 | $q \ll p(\cdot\|x)$일 때 weight collapse |
| Monotonic improvement | 실전 학습은 아닐 수 있음 (SNR, Rainforth) |
| Decoder와 encoder가 공동 이익 | Encoder는 $K$ 과도 시 악화 (정리 5.5) |

**실무 팁**:
- Evaluation: $K = 1000$~$10000$으로 log-likelihood 정확 추정
- Training: $K = 5$~$50$으로 balanced
- **DReG** (Doubly Reparameterized Gradient, Tucker 2019): encoder gradient 개선 변형

---

## 📌 핵심 정리

$$\boxed{\mathcal{L}_K = \mathbb{E}\left[\log\frac{1}{K}\sum_{k=1}^K w_k\right], \quad w_k = p(x, z_k)/q(z_k|x)}$$

$$\boxed{\mathcal{L}_1 \leq \mathcal{L}_2 \leq \cdots \to \log p(x)}$$

핵심:
- Jensen gap을 **$K$ 샘플로 감소**
- Monotonic in $K$ — 항상 tighter
- 수렴 rate $O(1/K)$
- 학습 시엔 **$K$ 과도 주의** (encoder gradient SNR)

---

## 🤔 생각해볼 문제

**문제 1** (기초): $K = 2$인 IWAE와 $K = 1$인 IWAE (= VAE)의 bound 차이를 간단 수식으로.

<details>
<summary>해설</summary>

$\mathcal{L}_1 = \mathbb{E}[\log w]$.

$\mathcal{L}_2 = \mathbb{E}[\log(\bar w_2)] = \mathbb{E}[\log\frac{w_1 + w_2}{2}]$.

Gap:
$\mathcal{L}_2 - \mathcal{L}_1 = \mathbb{E}[\log(w_1 + w_2)/2] - \mathbb{E}[\log w_1]$
$= \mathbb{E}[\log(1 + w_2/w_1)/2]$ ... 

양의 값 (Jensen). 작은 개선이지만 $K$ 증가로 누적.

실전: $K=5$ 정도면 이미 상당한 차이 (log-likelihood estimation 2~5 nats 더 tight).

</details>

**문제 2** (심화): IWAE의 **SNR 문제**(정리 5.5)를 완화하는 DReG estimator의 아이디어?

<details>
<summary>해설</summary>

**문제**: $\nabla_\phi \mathcal{L}_K$가 $\tilde w_k \cdot \nabla \log q(z_k|x)$ 형태. 이 $\tilde w_k$는 $\phi$에도 의존 → gradient가 복잡한 coupling.

**DReG** (Doubly Reparameterized Gradient, Tucker 2019):
- $z_k$뿐 아니라 $w_k$의 일부도 reparam
- Score function 기여를 감소, pathwise 기여를 증폭
- 결과: encoder gradient의 **SNR이 $K$와 함께 증가** (opposite of standard IWAE)

실전: IWAE-DReG가 VAE보다 **encoder도 더 나은** 학습 → ELBO 개선.

</details>

**문제 3** (AI 연결): IWAE는 $\log p(x)$를 정확히 계산하는 또 다른 방법인 **annealed importance sampling (AIS)**과 어떻게 비교되나?

<details>
<summary>해설</summary>

| | IWAE | AIS |
|---|---|---|
| Structure | Single-step $K$ samples | $T$-step annealed sequence |
| Bound | Lower (by $K$ samples) | Unbiased (in principle) |
| Cost | $O(K)$ forward passes | $O(T)$ MCMC-like steps |
| Tight | $O(1/K)$ | Exponentially $T$ |
| 구현 복잡도 | 간단 | 복잡 (intermediate distributions) |

**AIS**: $\pi_0 \to \pi_1 \to \cdots \to \pi_T = p(\cdot|x)$, 점진적으로 data distribution으로.

VAE evaluation 시 **AIS로 log-likelihood 정확 추정** (Wu et al. 2017). IWAE는 빠른 근사.

최근 Diffusion이 둘을 **통합** — forward process가 annealing, reverse가 importance-weighted inference.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 04. Amortized Inference](./04-amortized-inference.md) | [📚 README로 돌아가기](../README.md) | [Ch4-01. Metropolis-Hastings 재정리 ▶](../ch4-mcmc/01-metropolis-hastings.md) |

</div>
