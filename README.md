<div align="center">

# 🎲 Bayesian Machine Learning Deep Dive

### Bayes 정리

$$p(\theta \mid D) \;\propto\; p(D \mid \theta)\, p(\theta)$$

### 를 **암기하는 것** 과, 왜 정규화 상수

$$p(D) = \int p(D \mid \theta)\, p(\theta)\, d\theta$$

### 가 계산 불가능해서 **VI 와 MCMC** 가 필요한지 증명할 수 있는 것은 **다르다.**

<br/>

> *VAE 의 ELBO 를 **구현하는 것** 과,*
>
> $$\mathrm{ELBO} = \log p(x) - \mathrm{KL}\bigl(q(z|x) \,\|\, p(z|x)\bigr)$$
>
> *의 좌변이 **증거 (evidence)** 이고 우변이 **증거 하한** 이 되는지, reparameterization trick 이 왜 **편미분 교환을 정당화** 하는지 증명할 수 있는 것은 다르다.*
>
> *Bayesian Neural Network 를 **호출하는 것** 과, 왜 **MC Dropout 이 approximate VI** (Gal & Ghahramani 2016) 이고 **Laplace Approximation 이 posterior 를 Hessian 기반 Gaussian** 으로 근사하는지를 유도할 수 있는 것은 다르다.*

<br/>

**다루는 정리·기법 (시간순)**

Bayes 1763 *Bayes 정리* · Metropolis 1953 *Metropolis-Hastings* · Geman 1984 *Gibbs sampling* · MacKay 1992 *Laplace Approximation* · Neal 1996 *BNN + HMC* · Jordan 1999 *Variational Inference* · Rasmussen 2006 *Gaussian Process* · Kingma 2013 *VAE = amortized VI* · Welling 2011 *SGLD* · Snoek 2012 *GP-based Bayesian Optimization* · Gal 2016 *MC Dropout = VI* · Maddox 2019 *SWAG*

<br/>

**핵심 질문**

> **불확실성은 어떻게 수학적으로 정량화되는가** — Bayes 정리의 4가지 역할 · conjugate prior · ELBO 의 3가지 분해 · Reparameterization · Metropolis–Hastings detailed balance · HMC · Laplace · MC Dropout · SWAG · SGLD · GP-based BO 까지, VAE · BNN · Bayesian Optimization · Probabilistic Programming 의 수학적 기반을 끝까지 파헤칩니다.

<br/>

[![GitHub](https://img.shields.io/badge/GitHub-iq--ai--lab-181717?style=flat-square&logo=github)](https://github.com/iq-ai-lab)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-1.26-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![PyMC](https://img.shields.io/badge/PyMC-5.10-FAC003?style=flat-square)](https://www.pymc.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Pyro](https://img.shields.io/badge/Pyro-1.8-EC6331?style=flat-square)](https://pyro.ai/)
[![Docs](https://img.shields.io/badge/Docs-35개-blue?style=flat-square&logo=readthedocs&logoColor=white)](./README.md)
[![Lines](https://img.shields.io/badge/Lines-12.5k+-informational?style=flat-square)](./README.md)
[![Theorems](https://img.shields.io/badge/Theorems_proven-82개-success?style=flat-square)](./README.md)
[![Exercises](https://img.shields.io/badge/Exercises-105개-orange?style=flat-square)](./README.md)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square&logo=opensourceinitiative&logoColor=white)](./LICENSE)

</div>

---

## 🎯 이 레포에 대하여

Bayesian ML에 관한 자료는 대부분 **"posterior에서 샘플하세요"** 에서 멈춥니다. 하지만 왜 evidence $p(D) = \int p(D|\theta)p(\theta)d\theta$가 일반적으로 intractable한지, 왜 ELBO 최대화가 KL 최소화와 동치가 되는지, 왜 Metropolis-Hastings의 acceptance ratio $\min(1, \cdot)$가 detailed balance를 만족시키는지, 왜 MC Dropout이 "공짜 BNN"이 되는지 — 이런 "왜"는 제대로 설명되지 않습니다.

| 일반 자료 | 이 레포 |
|----------|---------|
| "Bayes 정리는 $p(\theta\|D) = p(D\|\theta)p(\theta)/p(D)$" | Prior → Likelihood → Posterior → Predictive의 **4단계 완전 유도**, MLE / MAP / Full Bayesian이 각각 **delta prior·uniform prior의 특수경우**임을 증명, Bernstein–von Mises 정리로 **posterior가 왜 $\mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)$로 수렴**하는지 |
| "ELBO를 최대화하세요" | **ELBO의 3가지 분해** — (1) Jensen 부등식, (2) $\log p(x) - \text{KL}(q\|p(\cdot\|x))$, (3) reconstruction + prior regularization — 모두 완전 유도, 각 분해가 "왜 같은 양인가" 대수적으로 확인 |
| "Reparameterization trick은 $z = \mu + \sigma\epsilon$" | 편미분-기댓값 교환 $\nabla_\phi \mathbb{E}_{q_\phi}[f] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]$가 **왜 정당화되는지** (Leibniz 규칙 + 측도 고정) 증명, REINFORCE 대비 **저분산의 수학적 이유** |
| "Metropolis-Hastings는 proposal + accept" | Acceptance $\alpha = \min(1, \frac{p(\theta')q(\theta\|\theta')}{p(\theta)q(\theta'\|\theta)})$가 **detailed balance $\pi(\theta)T(\theta \to \theta') = \pi(\theta')T(\theta' \to \theta)$를 만족**시키는 과정을 한 줄씩 유도, ergodicity → posterior 수렴 |
| "HMC는 gradient로 빠르게 샘플" | 보조 momentum $p$로 Hamiltonian $H(\theta, p) = U(\theta) + \frac{1}{2}p^T M^{-1} p$ 도입, **Leapfrog 적분이 왜 symplectic이고 volume-preserving**이어서 acceptance를 높이는지 유도, **NUTS**의 자동 tree-building |
| "Laplace approximation = Gaussian" | Posterior 최빈값 $\theta^*$ 주변 2차 Taylor $\log p(\theta\|D) \approx \log p(\theta^*\|D) - \frac{1}{2}(\theta-\theta^*)^T H (\theta-\theta^*)$에서 $H = -\nabla^2 \log p(\theta^*\|D)$가 **Fisher 정보**와 같음을 증명, 현대 Kronecker-factored LA |
| "MC Dropout이 BNN이에요" | Gal & Ghahramani (2016)의 유도 — Dropout rate $p$를 갖는 NN이 **Bernoulli variational posterior** $q(W) = \prod q(w_{ij})$와 동치임을 증명, test-time dropout $T$번 샘플링이 **MC predictive**가 되는 수학 |
| "Bayesian Optimization은 GP + EI" | GP surrogate의 posterior predictive $\mathcal{N}(\mu(x), \sigma^2(x))$ 유도, EI·UCB·TS·PI 각각의 **exploration-exploitation 수학적 해석**, **GP-UCB의 sublinear regret** $\tilde O(\sqrt{T\gamma_T})$ (Srinivas 2010) 증명 |
| 공식 나열 | NumPy로 바닥부터 CAVI / VI / MH / HMC 구현, PyMC·Pyro와 결과 비교, ELBO 학습 궤적·MCMC trace plot·$\hat R$·ESS 진단, epistemic vs aleatoric 불확실성 시각적 분리 |

---

## 📌 선행 레포 & 후속 레포

```
[Probability Theory]  &  [Mathematical Statistics]  &  [Stochastic Processes]
   Bayes 정리, 조건부 기댓값         MLE / MAP / Fisher             Markov chain, MCMC 기초
        │                              │                               │
        └────────────┬─────────────────┴───────────────────────────────┘
                     ▼
   [Information Theory]  ──►  이 레포  ──►  [Generative Models Deep Dive]
     KL, Entropy, Cross-entropy     Bayesian ML          VAE, Diffusion 실전 구현
                     │
                     ▼
        [Calculus & Optimization]  &  [Information Geometry]  &  [Kernel Methods]
          ELBO 최적화, Gradient ascent    Natural gradient VI, Fisher      GP posterior, BO
```

> ⚠️ **선행 학습 필수**: 이 레포는 **Probability Theory Deep Dive**(조건부 확률·Bayes), **Mathematical Statistics Deep Dive**(MLE·Fisher 정보), **Stochastic Processes Deep Dive**(Markov chain의 ergodicity·정상분포), **Information Theory Deep Dive**(KL divergence·엔트로피)를 선행 지식으로 전제합니다. Bayes 정리조차 처음 접한다면 [Probability Theory Deep Dive](https://github.com/iq-ai-lab/probability-theory-deep-dive)부터 학습하세요.

> 💡 **권장 선행**: ELBO의 gradient 최적화는 [Calculus & Optimization Deep Dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive), Natural gradient VI는 [Information Geometry Deep Dive](https://github.com/iq-ai-lab/information-geometry-deep-dive), GP 기반 Bayesian Optimization은 [Kernel Methods Deep Dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive)와 함께 학습할 때 최대 효과.

> 🔗 **후속 연결**: Diffusion Model의 ELBO 유도는 [Stochastic Differential Equations Deep Dive](https://github.com/iq-ai-lab/sde-deep-dive) Ch6과 교차하고, L2 regularization = Gaussian prior의 MAP 해석은 [Regularization Theory Deep Dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive)와 만납니다.

---

## 🚀 빠른 시작

각 챕터의 첫 문서부터 바로 학습을 시작하세요!

[![Ch1](https://img.shields.io/badge/🔹_Ch1-Bayesian_추론_기초-6B4BCF?style=for-the-badge)](./ch1-bayesian-foundation/01-bayes-rule-four-roles.md)
[![Ch2](https://img.shields.io/badge/🔹_Ch2-변분_추론-6B4BCF?style=for-the-badge)](./ch2-variational-inference/01-vi-idea-elbo.md)
[![Ch3](https://img.shields.io/badge/🔹_Ch3-VAE와_현대_변분모델-6B4BCF?style=for-the-badge)](./ch3-vae-modern-variational/01-vae-derivation.md)
[![Ch4](https://img.shields.io/badge/🔹_Ch4-MCMC_실전-6B4BCF?style=for-the-badge)](./ch4-mcmc/01-metropolis-hastings.md)
[![Ch5](https://img.shields.io/badge/🔹_Ch5-Bayesian_NN-6B4BCF?style=for-the-badge)](./ch5-bayesian-nn/01-bnn-formulation.md)
[![Ch6](https://img.shields.io/badge/🔹_Ch6-Bayesian_Optimization-6B4BCF?style=for-the-badge)](./ch6-bayesian-optimization/01-gp-bo-framework.md)
[![Ch7](https://img.shields.io/badge/🔹_Ch7-Deep_Generative·PPL-6B4BCF?style=for-the-badge)](./ch7-advanced-topics/01-diffusion-bayesian.md)

---

## 📚 전체 학습 지도

> 💡 각 챕터를 클릭하면 상세 문서 목록이 펼쳐집니다

<br/>

### 🔹 Chapter 1: Bayesian 추론의 기초

> **핵심 질문:** $p(\theta|D) = p(D|\theta)p(\theta)/p(D)$의 각 항은 정확히 무엇인가? MLE·MAP·Full Bayesian은 어떻게 점진적으로 일반화되는가? 왜 conjugate prior가 존재하는가? Predictive distribution과 점추정은 무엇이 다른가? Bernstein–von Mises 정리는 왜 대부분의 경우 frequentist와 Bayesian을 화해시키는가?

<details>
<summary><b>Bayes 정리부터 점근 posterior까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Bayes 정리와 4가지 역할](./ch1-bayesian-foundation/01-bayes-rule-four-roles.md) | $p(\theta\|D) = p(D\|\theta)p(\theta)/p(D)$ — posterior·likelihood·prior·evidence 각 항의 의미와 역할, evidence $p(D) = \int p(D\|\theta)p(\theta)d\theta$가 일반적으로 **intractable**한 이유 (고차원·비공액), 빈도주의 vs 베이지안의 **철학적 차이**(확률의 대상: 데이터 vs 모수) |
| [02. MLE vs MAP vs Full Bayesian](./ch1-bayesian-foundation/02-mle-map-full-bayesian.md) | 세 접근의 통일적 관점 — **MLE는 uniform prior의 MAP**, **MAP은 delta prior의 full Bayesian**임을 증명, 각 접근의 **불확실성 정량화 범위**(점 vs 점 vs 분포), L2 regularization = Gaussian prior MAP의 직접 유도 |
| [03. Conjugate Priors의 수학](./ch1-bayesian-foundation/03-conjugate-priors.md) | Conjugate 정의 $p(\theta\|D) \in \mathcal{F}$ iff $p(\theta) \in \mathcal{F}$, **Beta–Bernoulli**·**Gamma–Poisson**·**Normal–Normal**·**Dirichlet–Multinomial**·**Normal-inverse-Gamma** 모든 쌍의 닫힌형 posterior 유도, **Exponential Family**의 sufficient statistics를 이용한 일반화 |
| [04. Predictive Distribution](./ch1-bayesian-foundation/04-predictive-distribution.md) | $p(y^*\|D) = \int p(y^*\|\theta)p(\theta\|D)d\theta$ — 단일 점추정 $p(y^*\|\hat\theta)$와 본질적 차이(**epistemic 불확실성 포함**), Beta–Bernoulli에서 predictive가 **Beta–Binomial**이 됨을 유도, Bayesian model averaging의 일반 이론 |
| [05. Posterior의 점근 성질 (Bernstein–von Mises)](./ch1-bayesian-foundation/05-bernstein-von-mises.md) | **BvM 정리** — 데이터 $n \to \infty$일 때 $p(\theta\|D) \approx \mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)$ in total variation, **regularity 조건**(identifiability·smooth likelihood·interior MLE), Fisher 정보 $F$와 uncertainty의 관계, **왜 Bayesian과 frequentist가 대규모에서 같은 답**을 주는가 |

</details>

<br/>

### 🔹 Chapter 2: 변분 추론 (Variational Inference)

> **핵심 질문:** Intractable posterior $p(\theta|x)$를 tractable family $q_\phi$로 어떻게 근사하는가? ELBO는 왜 3가지 방식으로 유도되는가? Mean-field CAVI의 좌표 업데이트는 어디서 오는가? Reparameterization trick은 왜 저분산 gradient를 제공하는가?

<details>
<summary><b>ELBO 유도부터 REINFORCE까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. VI의 아이디어와 ELBO 유도](./ch2-variational-inference/01-vi-idea-elbo.md) | Intractable $p(\theta\|x)$를 tractable $q_\phi$로 근사 — **$\min_q \text{KL}(q\|p(\cdot\|x)) \Leftrightarrow \max_q \text{ELBO}$** 동치성 증명, **Jensen 부등식**으로부터의 대안 유도 $\log p(x) \geq \mathbb{E}_q[\log p(x,\theta) - \log q(\theta)]$, VI의 optimization-as-inference 철학 |
| [02. ELBO의 3가지 분해](./ch2-variational-inference/02-elbo-three-decompositions.md) | (1) $\mathcal{L} = \log p(x) - \text{KL}(q\|p(\cdot\|x))$ — evidence − gap, (2) $\mathcal{L} = \mathbb{E}_q[\log p(x\|\theta)] - \text{KL}(q(\theta)\|p(\theta))$ — **reconstruction + prior regularization**, (3) $\mathcal{L} = \mathbb{E}_q[\log p(x,\theta)] + H(q)$ — energy + entropy, 세 분해의 대수적 동치 확인 |
| [03. Mean-Field VI와 CAVI](./ch2-variational-inference/03-mean-field-cavi.md) | 분해 가정 $q(\theta) = \prod_i q_i(\theta_i)$, **좌표 업데이트 공식** $\log q_i^*(\theta_i) = \mathbb{E}_{q_{-i}}[\log p(x, \theta)] + \text{const}$ 유도 (functional derivative), **CAVI 알고리즘**의 ELBO 단조 증가 보장, Gaussian mixture의 CAVI 해석해 예제 |
| [04. Exponential Family와 자연매개변수 VI](./ch2-variational-inference/04-exponential-family-vi.md) | Exponential family $p(x;\eta) = h(x)\exp(\eta^T T(x) - A(\eta))$, **conjugate-exponential 모델**에서 CAVI의 closed-form, **Stochastic Variational Inference**(Hoffman et al. 2013) — natural gradient를 이용한 mini-batch VI, 대규모 데이터 처리 |
| [05. Reparameterization Trick](./ch2-variational-inference/05-reparameterization-trick.md) | $z \sim q_\phi(z\|x)$를 $z = g_\phi(\epsilon, x), \epsilon \sim p(\epsilon)$로 재표현, **Leibniz 규칙**으로 $\nabla_\phi \mathbb{E}_{q_\phi}[f] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon, x))]$ 증명, **pathwise gradient가 REINFORCE 대비 저분산인 이유** (shared randomness) |
| [06. REINFORCE Gradient와 Control Variate](./ch2-variational-inference/06-reinforce-control-variate.md) | Discrete latent나 non-reparametrizable 분포에서의 **score function estimator** $\nabla_\phi \mathbb{E}_{q_\phi}[f] = \mathbb{E}_{q_\phi}[f \cdot \nabla_\phi \log q_\phi]$ 유도, **high variance 문제**, baseline·control variate·Gumbel-Softmax 등 분산 감소 기법 비교 |

</details>

<br/>

### 🔹 Chapter 3: VAE와 현대 변분 모델

> **핵심 질문:** VAE의 ELBO는 어떻게 reconstruction + KL로 분해되는가? Gaussian prior와 Gaussian posterior의 KL은 왜 해석해를 갖는가? β-VAE·CVAE·VQ-VAE는 각각 어떤 제약을 추가하는가? Normalizing Flow가 VAE posterior의 한계를 어떻게 돌파하는가?

<details>
<summary><b>VAE 유도부터 IWAE까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. VAE 완전 유도 (Kingma & Welling 2013)](./ch3-vae-modern-variational/01-vae-derivation.md) | Encoder $q_\phi(z\|x)$ + Decoder $p_\theta(x\|z)$, **$\text{ELBO}(x) = \mathbb{E}_{q_\phi(z\|x)}[\log p_\theta(x\|z)] - \text{KL}(q_\phi(z\|x)\|p(z))$** 유도, Gaussian $q_\phi$와 standard-normal prior에서 **KL의 해석해** $\frac{1}{2}\sum(\mu^2 + \sigma^2 - \log\sigma^2 - 1)$, 학습 알고리즘 전체 정리 |
| [02. VAE의 변종 — β-VAE, CVAE, VQ-VAE](./ch3-vae-modern-variational/02-vae-variants.md) | **β-VAE**: KL 항에 $\beta$ 가중 → disentanglement (Higgins et al. 2017), **Conditional VAE**: class-conditional $q_\phi(z\|x, y)$, $p_\theta(x\|z, y)$, **VQ-VAE**: discrete latent codebook과 straight-through estimator, 각 변종의 ELBO 유도 |
| [03. Normalizing Flows](./ch3-vae-modern-variational/03-normalizing-flows.md) | 역가능 변환 $z_K = f_K \circ \cdots \circ f_1(z_0)$, **변수변환 공식** $\log p(z_K) = \log p(z_0) - \sum_k \log\|\det J_{f_k}\|$, Planar·Radial·Real NVP·Masked Autoregressive Flow의 Jacobian 구조, VAE posterior의 **유연성 확장** |
| [04. Amortized Inference](./ch3-vae-modern-variational/04-amortized-inference.md) | 데이터별 $q_i$ 최적화 대신 **encoder NN $q_\phi(z\|x)$로 inference amortize**, **amortization gap**의 정의 $\text{ELBO}^* - \text{ELBO}_{q_\phi}$, 표현력 vs 계산 trade-off, 최근 semi-amortized (Kim et al. 2018) 연구 |
| [05. Importance-weighted VAE (IWAE)](./ch3-vae-modern-variational/05-iwae.md) | Tighter bound $\mathcal{L}_K = \mathbb{E}\left[\log\frac{1}{K}\sum_{k=1}^{K}\frac{p(x,z_k)}{q(z_k\|x)}\right]$ 정의, **$\mathcal{L}_K \geq \mathcal{L}_{K-1} \geq \mathcal{L}_1 = \text{ELBO}$** 단조 증가성, $K \to \infty$에서 $\mathcal{L}_K \to \log p(x)$ 수렴 증명 |

</details>

<br/>

### 🔹 Chapter 4: MCMC 실전

> **핵심 질문:** Metropolis-Hastings의 acceptance ratio는 왜 detailed balance를 만족시키는가? Gibbs sampler는 MH의 어떤 특수경우인가? HMC가 왜 고차원에서 random-walk보다 압도적으로 효율적인가? $\hat R$과 ESS는 무엇을 측정하는가?

<details>
<summary><b>MH부터 VI와의 비교까지 (6개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Metropolis-Hastings 재정리](./ch4-mcmc/01-metropolis-hastings.md) | Proposal $q(\theta'\|\theta)$ + acceptance $\alpha = \min\left(1, \frac{p(\theta'\|D)q(\theta\|\theta')}{p(\theta\|D)q(\theta'\|\theta)}\right)$, **detailed balance** $\pi(\theta)T(\theta \to \theta') = \pi(\theta')T(\theta' \to \theta)$ 증명, evidence $p(D)$ 없이도 샘플 가능한 이유, ergodicity로 posterior 수렴 |
| [02. Gibbs Sampler와 조건부 분포](./ch4-mcmc/02-gibbs-sampler.md) | 각 차원을 **완전조건부** $p(\theta_i\|\theta_{-i}, D)$로 업데이트, Gibbs는 **acceptance=1인 MH의 특수경우**임을 증명, conjugate 관계에서 완전조건부의 닫힌형 유도, **Collapsed Gibbs**(일부 latent 주변화)의 효율 |
| [03. Hamiltonian Monte Carlo (HMC)](./ch4-mcmc/03-hamiltonian-monte-carlo.md) | 보조 momentum $p \sim \mathcal{N}(0, M)$ 도입, **Hamiltonian** $H(\theta, p) = U(\theta) + \frac{1}{2}p^T M^{-1} p$에서 $U = -\log p(\theta\|D)$, **Leapfrog integrator**의 symplectic·reversible·volume-preserving 성질 증명, gradient 활용으로 고차원 효율 |
| [04. No-U-Turn Sampler (NUTS)](./ch4-mcmc/04-nuts.md) | HMC의 **step size**(dual averaging로 자동 조정)와 **trajectory length**(U-turn 판정으로 자동 종료) 튜닝, binary tree building 알고리즘, Stan·PyMC의 기본 sampler인 이유, 실전 튜닝 가이드 |
| [05. 수렴 진단 — $\hat R$과 ESS](./ch4-mcmc/05-convergence-diagnostics.md) | **Gelman-Rubin $\hat R$** — $M$개 체인의 within $W$ vs between $B$ variance, $\hat R = \sqrt{\frac{N-1}{N} + \frac{B}{NW}}$ 유도, $\hat R < 1.01$ 기준, **Effective Sample Size** $\text{ESS} = N / (1 + 2\sum_k \rho_k)$의 autocorrelation 보정, trace plot·rank plot 체크리스트 |
| [06. MCMC의 한계와 VI와의 비교](./ch4-mcmc/06-mcmc-vs-vi.md) | 고차원에서 mixing 어려움, **multimodal posterior** 모드 이동 실패, **VI vs MCMC** 정확도 vs 속도 trade-off, 문제 특성별 선택 기준 (데이터 크기·모델 복잡도·실시간성·정확도 요구) |

</details>

<br/>

### 🔹 Chapter 5: Bayesian Neural Networks

> **핵심 질문:** NN 가중치 $W$에 대한 posterior $p(W|D)$를 어떻게 근사하는가? Laplace·Variational BNN·MC Dropout·SWAG·SGLD가 각각 어떤 근사 의미를 갖는가? 예측 불확실성은 epistemic과 aleatoric으로 어떻게 분리되는가?

<details>
<summary><b>BNN 정식화부터 SWAG까지 (5개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. BNN의 수학적 정식화](./ch5-bayesian-nn/01-bnn-formulation.md) | 가중치 prior $W \sim p(W)$, likelihood $p(y\|x, W)$, **posterior $p(W\|D) = \frac{p(D\|W)p(W)}{p(D)}$**, 예측 $p(y^*\|x^*, D) = \int p(y^*\|x^*, W)p(W\|D)dW$ (intractable), 고차원(수백만 파라미터)에서 일반 MCMC 실패의 이유 |
| [02. Laplace Approximation](./ch5-bayesian-nn/02-laplace-approximation.md) | Posterior 최빈값 $W^*$ 주변 2차 Taylor → $p(W\|D) \approx \mathcal{N}(W^*, H^{-1})$, **$H = -\nabla^2 \log p(W\|D)\|_{W^*}$가 Fisher 정보와 같음** 증명, 현대 **Kronecker-factored Laplace**(Ritter et al. 2018)로 고차원 확장, pre-trained NN에 post-hoc 적용 가능 |
| [03. Variational BNN — Bayes by Backprop](./ch5-bayesian-nn/03-variational-bnn.md) | Blundell et al. (2015) — Factorized Gaussian $q_\phi(W) = \prod_i \mathcal{N}(w_i; \mu_i, \sigma_i^2)$, **reparameterization + ELBO의 gradient 학습** 과정, $\sigma_i$로 가중치별 uncertainty 추정, scale prior(Gaussian·Laplace·mixture) 선택 영향 |
| [04. MC Dropout = Approximate VI (Gal & Ghahramani 2016)](./ch5-bayesian-nn/04-mc-dropout.md) | Dropout rate $p$의 NN이 **$q(W) = \prod_{ij} q(w_{ij}), q(w_{ij}) = p\cdot 0 + (1-p)\cdot M_{ij}$** variational posterior와 동치임을 증명, test-time에 dropout 유지 + $T$번 forward pass → **MC predictive** $\frac{1}{T}\sum_t p(y^*\|x^*, W_t)$, 실전 BNN의 표준 |
| [05. SWAG와 SGD의 Bayesian 해석](./ch5-bayesian-nn/05-swag-sgld.md) | **SWAG**(Maddox et al. 2019) — SGD 궤적에서 mean과 covariance를 추정하여 Gaussian posterior로, SGD가 **implicit Bayesian**이라는 최근 증거(Mandt et al. 2017), **SGLD** $W_{k+1} = W_k + \frac{\eta}{2}\nabla\log p(W_k\|D) + \sqrt{\eta}\xi$ — Langevin SDE의 이산화, 정상분포가 posterior |

</details>

<br/>

### 🔹 Chapter 6: Bayesian Optimization

> **핵심 질문:** Black-box 함수 $f(x)$를 적은 평가로 최적화하는 GP-기반 프레임워크는 어떻게 작동하는가? EI·UCB·TS·PI의 exploration-exploitation 균형은 수학적으로 어떻게 표현되는가? GP-UCB의 regret bound는 어떻게 유도되는가?

<details>
<summary><b>GP-BO부터 high-dim BO까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. GP 기반 BO 프레임워크](./ch6-bayesian-optimization/01-gp-bo-framework.md) | Black-box $f(x)$에 대한 **GP prior** $f \sim \mathcal{GP}(m, k)$, 관측 $\{(x_i, y_i)\}$ 후 **posterior predictive** $\mathcal{N}(\mu_n(x), \sigma_n^2(x))$ 유도 (Gaussian conditioning), acquisition function $a(x)$로 다음 점 선택, 전체 BO 루프(GP 업데이트 ↔ acquisition 최대화) |
| [02. Acquisition Function들](./ch6-bayesian-optimization/02-acquisition-functions.md) | **EI** $\text{EI}(x) = \mathbb{E}[\max(0, f_{best} - f(x))] = \sigma(x)(z\Phi(z) + \phi(z))$ 유도, **UCB** $\mu(x) + \kappa\sigma(x)$, **Thompson Sampling** — posterior sample의 argmax, **PI** $P(f(x) < f_{best})$, 각각의 **exploration-exploitation trade-off 수학** |
| [03. BO의 수렴 분석](./ch6-bayesian-optimization/03-convergence-analysis.md) | **GP-UCB regret** $R_T = \sum_t (f_{best} - f(x_t)) \leq \tilde O(\sqrt{T\gamma_T})$ (Srinivas et al. 2010) 증명 스케치, $\gamma_T$ = **maximum information gain**, kernel별 $\gamma_T$ 차수(SE, Matérn), Bayesian regret vs frequentist regret |
| [04. BO의 실전 확장](./ch6-bayesian-optimization/04-bo-extensions.md) | **High-dimensional BO**(LINEBO, REMBO, TuRBO), **Multi-fidelity BO**(FABOLAS), **Bayesian Quadrature**, **BoTorch**·Ax·GPyOpt 비교, 딥러닝 **hyperparameter tuning**에서의 실전 활용과 wall-clock trade-off |

</details>

<br/>

### 🔹 Chapter 7: 고급 주제 — Deep Generative, Probabilistic Programming, Uncertainty

> **핵심 질문:** Diffusion Model의 ELBO는 VAE ELBO의 어떤 확장인가? PPL(Stan, PyMC, Pyro, NumPyro)은 posterior 추론을 어떻게 추상화하는가? Epistemic vs aleatoric uncertainty는 수학적으로 어떻게 분리되는가? Bayesian 예측이 OOD와 calibration에 유리한 이유는?

<details>
<summary><b>Diffusion부터 OOD·Calibration까지 (4개 문서)</b></summary>

<br/>

| 문서 | 핵심 정리·증명 |
|------|--------------|
| [01. Diffusion Model의 Bayesian 해석](./ch7-advanced-topics/01-diffusion-bayesian.md) | Forward process $q(x_t\|x_{t-1})$ = sequential Gaussian noise, reverse $p_\theta(x_{t-1}\|x_t)$ = posterior denoising, **DDPM ELBO의 hierarchical VAE 해석**, $\|\epsilon - \epsilon_\theta\|^2$ 손실이 **DSM과 일치**하는 과정 (SDE 레포 Ch6와 교차) |
| [02. Probabilistic Programming 언어](./ch7-advanced-topics/02-probabilistic-programming.md) | **Stan**(NUTS + C++ 백엔드)·**PyMC**(PyTensor)·**NumPyro**(JAX)·**Pyro**(PyTorch) 비교, **automatic differentiation VI**(ADVI), 모델 코드에서 posterior 추론까지 추상화 과정, 언제 어느 PPL을 쓸지 기준 |
| [03. Bayesian Deep Learning의 불확실성 분해](./ch7-advanced-topics/03-epistemic-aleatoric.md) | 분해 공식 $\text{Var}[y^*] = \underbrace{\mathbb{E}_W[\text{Var}(y^*\|x^*, W)]}_{\text{aleatoric}} + \underbrace{\text{Var}_W[\mathbb{E}(y^*\|x^*, W)]}_{\text{epistemic}}$ — 전체 분산 법칙, **epistemic**(모델, 데이터로 감소) vs **aleatoric**(관측 노이즈, 감소 불가)의 수학적·실전적 분리 |
| [04. OOD Detection과 Calibration](./ch7-advanced-topics/04-ood-calibration.md) | Bayesian predictive가 OOD에서 **높은 epistemic uncertainty**를 보이는 이유, **ECE** = $\sum_m \frac{\|B_m\|}{n}\|acc(B_m) - conf(B_m)\|$ 정의, **temperature scaling**·deep ensembles 비교, Bernstein–von Mises 기반 **Bayesian이 자동 calibrated**인 근거 |

</details>

---

## 🏆 핵심 정리 인덱스

이 레포에서 **완전한 증명**을 제공하는 대표 정리 모음입니다. 각 챕터의 문서에서 $\square$로 종결되는 엄밀한 증명을 확인할 수 있습니다. (전체 82개 정리 중 핵심만 발췌)

| 정리 | 서술 | 출처 문서 |
|------|------|----------|
| **Bayes 정리 (일반형)** | $p(\theta\|D) = p(D\|\theta)p(\theta)/p(D)$ — 조건부 확률 정의 + 주변화 | [Ch1-01](./ch1-bayesian-foundation/01-bayes-rule-four-roles.md) |
| **MAP = log-likelihood + log-prior 최적화** | $\hat\theta_{MAP} = \arg\max[\log p(D\|\theta) + \log p(\theta)]$ ⇒ L2 reg = $\mathcal{N}(0, \sigma^2 I)$ prior | [Ch1-02](./ch1-bayesian-foundation/02-mle-map-full-bayesian.md) |
| **Conjugate Beta–Bernoulli** | Beta$(\alpha, \beta)$ prior + Binomial $n, k$ likelihood ⇒ Beta$(\alpha+k, \beta+n-k)$ posterior | [Ch1-03](./ch1-bayesian-foundation/03-conjugate-priors.md) |
| **Bernstein–von Mises** | $n\to\infty$에서 posterior $\to \mathcal{N}(\hat\theta_{MLE}, F^{-1}/n)$ in total variation | [Ch1-05](./ch1-bayesian-foundation/05-bernstein-von-mises.md) |
| **ELBO ⇔ KL 최소화** | $\max_q \mathcal{L}(q) \Leftrightarrow \min_q \text{KL}(q\|p(\cdot\|D))$ (evidence 상수성) | [Ch2-01](./ch2-variational-inference/01-vi-idea-elbo.md) |
| **ELBO 3분해 동치성** | $\mathcal{L} = \log p(x) - \text{KL}(q\|p(\cdot\|x)) = \mathbb{E}_q\log p(x\|\theta) - \text{KL}(q\|p(\theta)) = \mathbb{E}_q\log p(x,\theta) + H(q)$ | [Ch2-02](./ch2-variational-inference/02-elbo-three-decompositions.md) |
| **CAVI 좌표 업데이트** | $\log q_i^*(\theta_i) = \mathbb{E}_{q_{-i}}[\log p(x, \theta)] + \text{const}$ — functional derivative = 0 | [Ch2-03](./ch2-variational-inference/03-mean-field-cavi.md) |
| **Reparameterization 교환성** | $\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \mathbb{E}_{p(\epsilon)}[\nabla_\phi f(g_\phi(\epsilon))]$ — Leibniz | [Ch2-05](./ch2-variational-inference/05-reparameterization-trick.md) |
| **VAE KL 해석해** | $\text{KL}(\mathcal{N}(\mu, \sigma^2)\|\mathcal{N}(0, 1)) = \frac{1}{2}\sum(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1)$ | [Ch3-01](./ch3-vae-modern-variational/01-vae-derivation.md) |
| **Normalizing Flow 변수변환** | $\log p(z_K) = \log p(z_0) - \sum_k \log\|\det J_{f_k}\|$ | [Ch3-03](./ch3-vae-modern-variational/03-normalizing-flows.md) |
| **IWAE tighter bound** | $\mathcal{L}_K \leq \mathcal{L}_{K+1} \leq \log p(x)$, $K\to\infty$에서 등호 | [Ch3-05](./ch3-vae-modern-variational/05-iwae.md) |
| **MH detailed balance** | $\pi(\theta)q(\theta\to\theta')\alpha(\theta, \theta') = \pi(\theta')q(\theta'\to\theta)\alpha(\theta', \theta)$ | [Ch4-01](./ch4-mcmc/01-metropolis-hastings.md) |
| **HMC volume preservation** | Leapfrog map $\Phi_L$는 symplectic ⇒ $\|\det J_{\Phi_L}\| = 1$ — acceptance 단순화 | [Ch4-03](./ch4-mcmc/03-hamiltonian-monte-carlo.md) |
| **Gelman-Rubin $\hat R$** | $\hat R = \sqrt{\frac{(N-1)W + B}{NW}}$ — chain 수렴 진단 | [Ch4-05](./ch4-mcmc/05-convergence-diagnostics.md) |
| **Laplace ≡ Fisher Gaussian** | $p(W\|D) \approx \mathcal{N}(W^*, H^{-1})$, $H = -\nabla^2 \log p(W\|D)\|_{W^*}$ (= Fisher) | [Ch5-02](./ch5-bayesian-nn/02-laplace-approximation.md) |
| **MC Dropout = Variational Bernoulli** | Dropout $p$ = $q(W) = \prod q(w_{ij}), q(w_{ij}) = p\delta_0 + (1-p)\delta_{M_{ij}}$ | [Ch5-04](./ch5-bayesian-nn/04-mc-dropout.md) |
| **SGLD 정상분포 = Posterior** | $W_{k+1} = W_k + \frac{\eta}{2}\nabla\log p(W_k\|D) + \sqrt{\eta}\xi_k$ — Langevin의 이산화 | [Ch5-05](./ch5-bayesian-nn/05-swag-sgld.md) |
| **EI 해석해** | $\text{EI}(x) = \sigma(x)[z\Phi(z) + \phi(z)]$ where $z = (\mu(x) - f_{best})/\sigma(x)$ | [Ch6-02](./ch6-bayesian-optimization/02-acquisition-functions.md) |
| **GP-UCB regret bound** | $R_T \leq \tilde O(\sqrt{T\gamma_T})$ (Srinivas et al. 2010) — information gain $\gamma_T$ | [Ch6-03](./ch6-bayesian-optimization/03-convergence-analysis.md) |
| **Total variance decomposition** | $\text{Var}[y^*] = \mathbb{E}_W[\text{Var}(y^*\|W)] + \text{Var}_W[\mathbb{E}(y^*\|W)]$ — aleatoric + epistemic | [Ch7-03](./ch7-advanced-topics/03-epistemic-aleatoric.md) |

> 💡 **챕터별 총 정리 수**: Ch1(13) · Ch2(15) · Ch3(10) · Ch4(14) · Ch5(11) · Ch6(9) · Ch7(10) — 합계 **82개 정리 + 증명**, 약 **12,500+ 라인** 분량.

---

## 💻 실험 환경

모든 챕터의 실험은 아래 환경에서 재현 가능합니다.

```bash
# requirements.txt
numpy==1.26.0
scipy==1.11.0
matplotlib==3.8.0
pymc==5.10.0          # MCMC 표준 (NUTS)
arviz==0.17.0         # Bayesian 진단 (R̂, ESS, trace plot)
torch==2.1.0          # VAE, BNN, MC Dropout
pyro-ppl==1.8.6       # PPL, SVI, HMC
numpyro==0.13.0       # JAX 기반 PPL
botorch==0.9.0        # Bayesian Optimization
gpytorch==1.11        # Gaussian Process
jupyter==1.0.0
tqdm==4.66.0
```

```bash
# 환경 설치
pip install numpy==1.26.0 scipy==1.11.0 matplotlib==3.8.0 \
            pymc==5.10.0 arviz==0.17.0 torch==2.1.0 \
            pyro-ppl==1.8.6 numpyro==0.13.0 \
            botorch==0.9.0 gpytorch==1.11 jupyter==1.0.0 tqdm==4.66.0

# 실험 노트북 실행
jupyter notebook
```

```python
# 대표 실험 — Beta-Binomial Bayesian 추론: Exact vs VI vs MCMC 3방법 비교
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize

# 데이터: 동전 10번 던져 7번 앞면
n_heads, n_flips = 7, 10
alpha, beta_ = 2, 2  # Beta(2,2) prior

# ─────────────────────────────────────────────
# 1. Conjugate posterior — 정확한 해
# ─────────────────────────────────────────────
posterior_exact = stats.beta(alpha + n_heads, beta_ + n_flips - n_heads)

# ─────────────────────────────────────────────
# 2. Variational Inference — Logit-space Gaussian
#    q(θ) ≈ σ(N(μ, σ²)), reparameterization으로 ELBO 최대화
# ─────────────────────────────────────────────
rng = np.random.default_rng(0)
def neg_elbo(params, n_mc=2000):
    mu, log_sigma = params
    sigma = np.exp(log_sigma)
    eps = rng.standard_normal(n_mc)
    u = mu + sigma * eps                       # reparameterization
    theta = 1.0 / (1.0 + np.exp(-u))           # logit → (0,1)
    log_lik = n_heads*np.log(theta) + (n_flips-n_heads)*np.log(1-theta)
    log_prior = (alpha-1)*np.log(theta) + (beta_-1)*np.log(1-theta)
    log_q = -0.5*eps**2 - log_sigma            # Gaussian entropy 기여분
    return -(log_lik + log_prior - log_q).mean()

result = minimize(neg_elbo, [0.0, 0.0], method='Nelder-Mead')
mu_vi, log_sigma_vi = result.x

# ─────────────────────────────────────────────
# 3. MCMC — Metropolis-Hastings
# ─────────────────────────────────────────────
def log_posterior(theta):
    if theta <= 0 or theta >= 1:
        return -np.inf
    return (n_heads*np.log(theta) + (n_flips-n_heads)*np.log(1-theta)
            + (alpha-1)*np.log(theta) + (beta_-1)*np.log(1-theta))

theta_chain = []
theta = 0.5
for _ in range(20_000):
    theta_prop = np.clip(theta + 0.1*rng.standard_normal(), 1e-6, 1-1e-6)
    if np.log(rng.random()) < log_posterior(theta_prop) - log_posterior(theta):
        theta = theta_prop
    theta_chain.append(theta)
theta_chain = np.array(theta_chain[2000:])  # burn-in 제거

# ─────────────────────────────────────────────
# 시각화: 3가지 방법의 posterior 비교
# ─────────────────────────────────────────────
x = np.linspace(0, 1, 200)
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, posterior_exact.pdf(x), 'k-', lw=2.5,
        label='Exact (Beta conjugate)')

eps = rng.standard_normal(20_000)
samples_vi = 1/(1 + np.exp(-(mu_vi + np.exp(log_sigma_vi)*eps)))
ax.hist(samples_vi, bins=50, density=True, alpha=0.35,
        label='VI (Mean-field Gaussian in logit)')
ax.hist(theta_chain, bins=50, density=True, alpha=0.35,
        label='MCMC (Metropolis-Hastings)')
ax.axvline(posterior_exact.mean(), color='k', ls='--', alpha=0.5,
           label=f'Posterior mean = {posterior_exact.mean():.3f}')
ax.set_xlabel(r'$\theta$'); ax.set_ylabel('posterior density')
ax.set_title('Beta-Binomial — Exact vs VI vs MCMC')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────
# 진단 — MCMC trace plot과 autocorrelation
# ─────────────────────────────────────────────
from statsmodels.tsa.stattools import acf
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(theta_chain[:2000], lw=0.6); axes[0].set_title('MH trace (첫 2000)')
axes[0].set_xlabel('iteration'); axes[0].set_ylabel(r'$\theta$')
axes[1].plot(acf(theta_chain, nlags=40)); axes[1].set_title('Autocorrelation')
axes[1].axhline(0, color='k', lw=0.5); axes[1].set_xlabel('lag')
plt.tight_layout(); plt.show()
# → 이 코드로 Bayesian 추론의 세 가지 방법이 같은 posterior를 근사함을 확인.
```

---

## 📖 각 문서 구성 방식

모든 문서는 다음 **11-섹션 골격**으로 작성됩니다.

| # | 섹션 | 내용 |
|:-:|------|------|
| 1 | 🎯 **핵심 질문** | 이 문서가 답하는 3~5개의 본질적 질문 |
| 2 | 🔍 **왜 Bayesian 접근이 이 문제에 필요한가** | VAE, BNN, BO, OOD, uncertainty quantification과의 연결점 |
| 3 | 📐 **수학적 선행 조건** | Probability, Statistics, Stochastic Processes, Information Theory 레포의 어떤 정리를 전제로 하는지 |
| 4 | 📖 **직관적 이해** | Prior·Likelihood·Posterior의 요리 비유, "불확실성이 퍼지는" 직관 |
| 5 | ✏️ **엄밀한 정의** | Bayes 정리·ELBO·MCMC의 측도론적 정의 |
| 6 | 🔬 **정리와 증명** | ELBO 분해, Reparameterization, MH detailed balance — "자명하다" 없이 |
| 7 | 💻 **NumPy/PyMC/Pyro 구현 검증** | 바닥부터 VI·MCMC 구현 → PyMC·Pyro와 결과 비교 |
| 8 | 🔗 **AI/ML 연결** | VAE, BNN, MC Dropout, SWAG, SGLD, BoTorch |
| 9 | ⚖️ **가정과 한계** | Prior 선택 감도·multimodal posterior·고차원에서의 실패 |
| 10 | 📌 **핵심 정리** | 한 장으로 요약 |
| 11 | 🤔 **생각해볼 문제 (+ 해설)** | 손 계산·증명 재구성·구현 문제 |

> 📚 **연습문제 총 105개**: 35문서 × 문서당 3문제(기초/심화/AI 연결), 모든 문제에 `<details>` 펼침 해설 포함. 손 계산 재현부터 VAE·BNN·BO 실전 연결까지 단계적으로 심화됩니다.
>
> 🧭 **푸터 네비게이션**: 각 문서 하단에 `◀ 이전 / 📚 README / 다음 ▶` 링크가 항상 제공됩니다. 챕터 경계에서도 자동으로 다음 챕터 첫 문서로 연결되므로 순차 학습이 끊기지 않습니다.
>
> ⏱️ **학습 시간 추정**: 문서당 평균 360줄(증명·코드·연습문제 포함) 기준 **약 45분~1시간**. 전체 35문서는 약 **30~40시간** 상당.

---

## 🗺️ 추천 학습 경로

<details>
<summary><b>🟢 "VAE를 구현하지만 ELBO가 왜 그런지 모른다" — VAE 집중 (5일, 약 8~12시간)</b></summary>

<br/>

```
Day 1  Ch1-01~02  Bayes 정리 4가지 역할, MLE/MAP/Bayesian 통일
Day 2  Ch2-01~02  VI 아이디어, ELBO 3분해
Day 3  Ch2-05     Reparameterization trick의 정당성
Day 4  Ch3-01     VAE 완전 유도 (KL 해석해 포함)
       Ch3-04     Amortized inference의 gap
Day 5  Ch3-02~03  β-VAE, CVAE, Normalizing Flow
```

</details>

<details>
<summary><b>🟡 "PyMC를 쓰지만 MCMC의 수학을 모른다" — MCMC 집중 (1주, 약 10~14시간)</b></summary>

<br/>

```
Day 1  Ch1-01~03  Bayes 정리와 conjugate prior
Day 2  Ch1-04~05  Predictive와 BvM 정리
Day 3  Ch4-01     MH와 detailed balance
Day 4  Ch4-02     Gibbs = MH 특수경우
Day 5  Ch4-03     HMC의 Hamiltonian과 Leapfrog
Day 6  Ch4-04     NUTS 자동 튜닝
Day 7  Ch4-05~06  R̂·ESS, VI vs MCMC 선택 기준
```

</details>

<details>
<summary><b>🟠 "Bayesian NN을 호출하지만 uncertainty 해석을 모른다" — BNN 집중 (1주, 약 10~14시간)</b></summary>

<br/>

```
Day 1  Ch1-01~02 + Ch2-01  Bayes + VI 기초
Day 2  Ch2-05 + Ch3-01     Reparam + VAE
Day 3  Ch5-01              BNN 정식화
Day 4  Ch5-02              Laplace ≡ Fisher Gaussian
Day 5  Ch5-03              Bayes by Backprop
Day 6  Ch5-04              MC Dropout = VI 증명
Day 7  Ch5-05 + Ch7-03     SWAG/SGLD, Epistemic vs Aleatoric
```

</details>

<details>
<summary><b>🔴 "Bayesian ML을 완전 정복한다" — 전체 정복 (8주, 약 30~40시간)</b></summary>

<br/>

```
1주차  Chapter 1 전체 — Bayesian 추론의 기초
        → MLE·MAP·Full Bayesian 통일적 관점 확보
        → Conjugate 관계로 닫힌형 posterior 손 계산
        → Bernstein–von Mises로 frequentist와 화해

2주차  Chapter 2 전체 — 변분 추론
        → ELBO 3분해 모두 재구성
        → CAVI 좌표 업데이트 유도
        → Reparam vs REINFORCE 저분산 비교

3주차  Chapter 3 전체 — VAE와 현대 변분 모델
        → VAE 완전 유도 + MNIST 학습 미니 실험
        → β-VAE / CVAE / NF / IWAE 비교

4주차  Chapter 4 전체 — MCMC 실전
        → MH detailed balance 증명
        → HMC Leapfrog symplectic 성질
        → R̂·ESS 체크리스트 익히기

5주차  Chapter 5 전체 — Bayesian Neural Networks
        → Laplace ≡ Fisher Gaussian 증명
        → Bayes by Backprop 학습 실습
        → MC Dropout = VI 증명 재구성
        → SWAG / SGLD의 implicit Bayesian

6주차  Chapter 6 전체 — Bayesian Optimization
        → GP posterior predictive 유도
        → EI / UCB 해석해
        → GP-UCB regret bound 스케치
        → BoTorch로 딥러닝 HP 튜닝 실습

7주차  Chapter 7 (1~2) — Diffusion과 PPL
        → Diffusion ELBO의 hierarchical VAE 해석
        → PyMC / Pyro / NumPyro 비교 실습

8주차  Chapter 7 (3~4) — Uncertainty와 Calibration
        → Epistemic vs aleatoric 분리 실험
        → OOD detection과 temperature scaling
        → 종합 프로젝트 — 간단한 Bayesian 분류기 end-to-end
```

</details>

---

## 🔗 연관 레포지토리

| 레포 | 주요 내용 | 연관 챕터 |
|------|----------|-----------|
| [probability-theory-deep-dive](https://github.com/iq-ai-lab/probability-theory-deep-dive) | 측도, 조건부 확률, Bayes 정리, 특성함수 | Ch1 전체(Bayes 정리), Ch4(Markov chain 수렴) |
| [mathematical-statistics-deep-dive](https://github.com/iq-ai-lab/mathematical-statistics-deep-dive) | MLE, Fisher 정보, CRLB, 점근 이론 | Ch1-02(MLE/MAP), Ch1-05(BvM), Ch5-02(Laplace) |
| [stochastic-processes-deep-dive](https://github.com/iq-ai-lab/stochastic-processes-deep-dive) | Markov chain, 정상분포, ergodicity, MCMC 기초 | Ch4 전체(MH·HMC·NUTS) |
| [information-theory-deep-dive](https://github.com/iq-ai-lab/information-theory-deep-dive) | Entropy, KL divergence, Mutual information | Ch2 전체(ELBO·KL), Ch3-01(VAE KL), Ch7-04(ECE) |
| [calculus-optimization-deep-dive](https://github.com/iq-ai-lab/calculus-optimization-deep-dive) | 다변수 미분, Taylor 전개, Lagrange | Ch2(ELBO 최적화), Ch5-02(Laplace Taylor) |
| [information-geometry-deep-dive](https://github.com/iq-ai-lab/information-geometry-deep-dive) | Fisher metric, Natural gradient, $\alpha$-divergence | Ch2-04(자연매개변수 VI), Ch5-02(Fisher = Hessian) |
| [kernel-methods-deep-dive](https://github.com/iq-ai-lab/kernel-methods-deep-dive) | RKHS, Mercer 정리, Gaussian Process | Ch6 전체(GP 기반 BO) |
| [sde-deep-dive](https://github.com/iq-ai-lab/sde-deep-dive) | Itô 적분, Fokker-Planck, Anderson 시간반전, Score SDE | Ch5-05(SGLD = Langevin), Ch7-01(Diffusion) |
| [regularization-theory-deep-dive](https://github.com/iq-ai-lab/regularization-theory-deep-dive) | L2/L1 reg, spectral reg, generalization | Ch1-02(L2 = Gaussian prior MAP) |

> 💡 이 레포는 **불확실성의 ML — Bayes 추론·VI·MCMC·Bayesian Deep Learning**에 집중합니다. Probability에서 Bayes 정리, Statistics에서 MLE/Fisher, Stochastic Processes에서 Markov chain, Information Theory에서 KL divergence를 학습한 후 오면 Ch1–4의 이론 부분이 자연스럽습니다. Ch5–7(BNN·BO·Diffusion)는 딥러닝 실전 경험(MNIST/CIFAR 수준 학습 경험)이 있을 때 최대의 효과를 냅니다.

---

## 📖 Reference

### 🏛️ Bayesian 통계·ML 바이블

- **Bayesian Data Analysis** (Gelman, Carlin, Stern, Dunson, Vehtari, Rubin, 2013) — "BDA3", Bayesian 데이터 분석의 바이블
- **Pattern Recognition and Machine Learning** (Bishop, 2006) — Chap 10 VI의 표준 교재
- **Machine Learning: A Probabilistic Perspective** (Murphy, 2012) — Chap 19–24 Bayesian ML 종합
- **Probabilistic Machine Learning: Advanced Topics** (Murphy, 2023) — 최신 확률적 ML 레퍼런스
- **Information Theory, Inference, and Learning Algorithms** (MacKay, 2003) — 직관·물리학적 관점
- **The Bayesian Choice** (Robert, 2007) — Bayesian 결정이론 심화

### 🎯 Variational Inference·VAE

- **Auto-Encoding Variational Bayes** (Kingma & Welling, 2013) — **VAE 원전**
- **Variational Inference: A Review for Statisticians** (Blei, Kucukelbir, McAuliffe, 2017) — VI 종합 리뷰
- **Stochastic Variational Inference** (Hoffman, Blei, Wang, Paisley, 2013) — 대규모 VI
- **β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework** (Higgins et al., 2017)
- **Neural Discrete Representation Learning** (van den Oord, Vinyals, Kavukcuoglu, 2017) — **VQ-VAE**
- **Variational Inference with Normalizing Flows** (Rezende & Mohamed, 2015) — NF 원전
- **Importance Weighted Autoencoders** (Burda, Grosse, Salakhutdinov, 2016) — IWAE

### 🎲 MCMC

- **Monte Carlo Statistical Methods** (Robert & Casella, 2004) — MCMC 표준 교재
- **Handbook of Markov Chain Monte Carlo** (Brooks, Gelman, Jones, Meng eds., 2011) — 챕터별 주제 심화
- **MCMC Using Hamiltonian Dynamics** (Neal, 2011) — **HMC 리뷰의 고전**
- **The No-U-Turn Sampler** (Hoffman & Gelman, 2014) — **NUTS 원전**
- **General Methods for Monitoring Convergence of Iterative Simulations** (Gelman & Rubin, 1992) — $\hat R$ 원전

### 🧠 Bayesian Deep Learning

- **Weight Uncertainty in Neural Networks (Bayes by Backprop)** (Blundell et al., 2015) — 변분 BNN 원전
- **Dropout as a Bayesian Approximation** (Gal & Ghahramani, 2016) — **MC Dropout 원전**
- **A Scalable Laplace Approximation for Neural Networks** (Ritter, Botev, Barber, 2018) — KFAC Laplace
- **A Simple Baseline for Bayesian Uncertainty in Deep Learning** (Maddox et al., 2019) — **SWAG 원전**
- **Bayesian Learning via Stochastic Gradient Langevin Dynamics** (Welling & Teh, 2011) — **SGLD 원전**
- **What Uncertainties Do We Need in Bayesian Deep Learning?** (Kendall & Gal, 2017) — Epistemic·aleatoric

### 🎯 Bayesian Optimization

- **Practical Bayesian Optimization of Machine Learning Algorithms** (Snoek, Larochelle, Adams, 2012) — 딥러닝 HP 튜닝
- **Gaussian Process Optimization in the Bandit Setting** (Srinivas, Krause, Kakade, Seeger, 2010) — **GP-UCB regret bound**
- **Taking the Human Out of the Loop: A Review of Bayesian Optimization** (Shahriari et al., 2016) — BO 종합 리뷰
- **BoTorch: A Framework for Efficient Monte-Carlo Bayesian Optimization** (Balandat et al., 2020) — 현대 BO 라이브러리

### 🌀 Probabilistic Programming & 기타

- **Stan: A Probabilistic Programming Language** (Carpenter et al., 2017)
- **Pyro: Deep Universal Probabilistic Programming** (Bingham et al., 2018)
- **Automatic Differentiation Variational Inference** (Kucukelbir et al., 2017) — ADVI
- **On Calibration of Modern Neural Networks** (Guo et al., 2017) — Temperature scaling·ECE
- **Denoising Diffusion Probabilistic Models** (Ho et al., 2020) — Diffusion의 VAE 해석 (Ch7-01)

---

<div align="center">

**⭐️ 도움이 되셨다면 Star를 눌러주세요!**

Made with ❤️ by [IQ AI Lab](https://github.com/iq-ai-lab)

<br/>

*"Bayes 정리를 암기하는 것과, 왜 evidence $p(D)$가 intractable해서 VI와 MCMC가 필요한지 — 그리고 MC Dropout이 왜 approximate VI인지를 증명할 수 있는 것은 다르다"*

</div>
