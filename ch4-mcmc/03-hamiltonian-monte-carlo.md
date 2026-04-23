# 03. Hamiltonian Monte Carlo (HMC)

## 🎯 핵심 질문

- 보조 momentum $p$를 도입한 **Hamiltonian** $H(\theta, p) = U(\theta) + \frac{1}{2}p^T M^{-1} p$의 아이디어는?
- **Leapfrog integrator**가 왜 **symplectic**·reversible·volume-preserving인가?
- HMC가 random-walk MH보다 **고차원에서** 왜 압도적으로 효율적인가?
- Potential $U = -\log p(\theta|D)$의 gradient가 어떻게 proposal을 만드는가?

---

## 🔍 왜 Bayesian 접근이 이 문제에 필요한가

**NUTS**(Stan, PyMC의 기본)의 엔진. BNN(Ch5) 같은 고차원에서 random-walk MH가 절망적으로 느릴 때 HMC가 유일한 실용 해결. **Leapfrog**의 대칭성이 posterior 샘플링의 정확도를 보장. **Hamiltonian dynamics**는 물리학·기하학의 deep connection — statistical manifold의 기하학(Ch5-02 Fisher, Information Geometry).

---

## 📐 수학적 선행 조건

- [Ch4-01 MH](./01-metropolis-hastings.md)
- 해밀턴 역학: $\dot\theta = \nabla_p H, \dot p = -\nabla_\theta H$
- Symplectic geometry 기초: volume preservation
- Auto-diff for $\nabla\log p$

---

## 📖 직관적 이해

### 왜 Hamiltonian?

Posterior $\pi(\theta) \propto e^{-U(\theta)}$ (potential $U = -\log\pi$). Random-walk은 지역 무작위 탐색 — 고차원에서 비효율.

HMC 아이디어: **momentum $p$를 도입**하고 Hamiltonian dynamics로 $\theta$를 이동. Momentum이 **"관성"을 제공** → 한 방향으로 길게 이동 → 독립 샘플로 빠르게.

### Joint canonical distribution

$(\theta, p)$의 joint:
$$\pi_{\text{joint}}(\theta, p) \propto \exp(-H(\theta, p)) = \exp(-U(\theta))\cdot\exp(-\frac{1}{2}p^T M^{-1}p)$$

$\theta$에 대해 marginalize하면 $\pi(\theta)$ 회복. 따라서 joint에서 샘플한 뒤 $p$를 버리면 $\theta \sim \pi$.

### 알고리즘 개요

```
Given θ_current:
1. Sample p ~ N(0, M)
2. Simulate Hamiltonian dynamics for L steps with step size ε
   using Leapfrog (θ, p) → (θ*, p*)
3. Accept (θ*, p*) with probability
   α = min(1, exp(H(θ, p) - H(θ*, p*)))
4. Return θ*
```

이상적 dynamics는 $H$ 보존 → $\alpha = 1$. Discretization 오차만 보정.

### 요리 비유

- Random-walk: "눈 감고 한 발 움직임"
- HMC: "앞 방향(gradient)으로 **몇 걸음 밀고** — momentum이 terrain을 따라 흐름"

고차원 spaghetti-bowl posterior에서 HMC가 바로 "그릇 방향"을 따라 탐색.

---

## ✏️ 엄밀한 정의

### 정의 3.1 — Hamiltonian

$$H(\theta, p) = U(\theta) + K(p), \quad U(\theta) = -\log\pi(\theta), \quad K(p) = \frac{1}{2}p^T M^{-1} p$$

- $U$: potential (data와 prior)
- $K$: kinetic energy (Gaussian)
- $M$: mass matrix (보통 $I$)

### 정의 3.2 — Hamiltonian Equations

$$\dot\theta = \nabla_p H = M^{-1} p, \qquad \dot p = -\nabla_\theta H = -\nabla U(\theta) = \nabla\log\pi(\theta)$$

### 정의 3.3 — Leapfrog Integrator

Step size $\epsilon$으로 이산화:

$$\begin{aligned}
p_{t+\epsilon/2} &= p_t - \frac{\epsilon}{2}\nabla U(\theta_t)\\
\theta_{t+\epsilon} &= \theta_t + \epsilon M^{-1} p_{t+\epsilon/2}\\
p_{t+\epsilon} &= p_{t+\epsilon/2} - \frac{\epsilon}{2}\nabla U(\theta_{t+\epsilon})
\end{aligned}$$

**Half-step momentum → full-step position → half-step momentum**.

### 정의 3.4 — HMC Proposal

$L$ Leapfrog steps 적용하여 $(\theta_0, p_0) \to (\theta_L, p_L)$. Acceptance:

$$\alpha = \min(1, \exp(H(\theta_0, p_0) - H(\theta_L, p_L)))$$

---

## 🔬 정리와 증명

### 정리 3.1 — Leapfrog는 Reversible (시간반전 대칭)

**명제**: Leapfrog의 역변환 = momentum 부호 뒤집은 forward leapfrog.

**증명**: 각 half-step을 back-propagation하면 정확히 reverse가 나옴. 더 정확히, forward $(θ, p) \to (θ', p')$에 이어 $(θ', -p')$에서 forward 하면 $(θ, -p)$ 돌아옴. $\square$

**귀결**: Detailed balance 검증에 reversible proposal이 필요 → MH acceptance로 반영 가능.

### 정리 3.2 — Leapfrog는 Volume-Preserving (Symplectic)

**명제**: Leapfrog Jacobian의 determinant가 1:

$$|\det J_{\text{Leapfrog}}(\theta, p)| = 1$$

**증명** (sketch):

각 half-step을 개별로 분석:
- $p$ half-step: $(\theta, p) \to (\theta, p - \frac{\epsilon}{2}\nabla U)$. Jacobian upper-triangular, diag = 1 → det = 1.
- $\theta$ full-step: $(\theta, p) \to (\theta + \epsilon M^{-1}p, p)$. Lower-triangular Jacobian, diag = 1 → det = 1.
- Second $p$ half: 마찬가지.

Composition: det = 1 · 1 · 1 = 1. $\square$

**귀결**: MH acceptance에서 Jacobian factor가 1 → 단순한 $\exp(\Delta H)$.

### 정리 3.3 — HMC Acceptance Simplification

**명제**: Leapfrog (reversible, volume-preserving) 사용 시:

$$\alpha(\theta, p, \theta', p') = \min(1, \exp(H(\theta, p) - H(\theta', p')))$$

**증명**:

일반 MH acceptance에서 proposal density가 $q(\cdot|\cdot)$. Leapfrog가 deterministic이므로 $q = \delta$, 하지만 reversibility와 volume preservation으로 Jacobian = 1. 결과:

$$\alpha = \min\left(1, \frac{\pi_{\text{joint}}(\theta', p')}{\pi_{\text{joint}}(\theta, p)}\right) = \min(1, e^{-H(\theta',p') + H(\theta, p)})$$

$\square$

### 정리 3.4 — 이상적 Dynamics는 $H$ 보존

**명제**: Continuous Hamiltonian equations의 해를 따라 $H$는 constant:

$$\frac{dH}{dt} = \nabla_\theta H \cdot \dot\theta + \nabla_p H \cdot \dot p = \nabla_\theta H \cdot \nabla_p H + \nabla_p H \cdot (-\nabla_\theta H) = 0$$

**증명**: 위 계산 직접. $\square$

**귀결**: 이상적으론 $\alpha = 1$. Discretization 오차만 작은 $\Delta H$ 생성 — acceptance로 수정.

### 정리 3.5 — Leapfrog의 수렴 차수

**명제**: Leapfrog는 **2차 symplectic integrator**:

$$H(\theta_L, p_L) - H(\theta_0, p_0) = O(\epsilon^2) \cdot L$$

(long-time $O(\epsilon^2)$ 유지, non-drift).

**실전 영향**: $\epsilon$를 절반으로 줄이면 $\Delta H$ 오차 $1/4$배 → acceptance rate 급상승.

### 정리 3.6 — HMC의 Dimension-Independent Mixing (Idealization)

**명제** (Beskos et al. 2013): 적절한 scaling 하에서 HMC의 optimal acceptance rate:

$$\alpha^*_{HMC} \approx 0.65$$

(vs random-walk $0.234$, Gibbs $1.0$).

그리고 **mixing time per dimension**이 dimension에 **준일정** — 고차원에서 대폭 개선.

---

## 💻 NumPy 구현

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)

# ────────────────────────────────────────────────
# Target: 2D banana-shaped posterior (strongly curved)
# π(θ) ∝ exp(-U(θ)), U = θ₁²/2 + (θ₂ + θ₁² - 2)²/2
# ────────────────────────────────────────────────
def U(theta):
    return 0.5*theta[0]**2 + 0.5*(theta[1] + theta[0]**2 - 2)**2

def grad_U(theta):
    # ∂U/∂θ1 = θ1 + 2θ1(θ2 + θ1² - 2)
    # ∂U/∂θ2 = θ2 + θ1² - 2
    g1 = theta[0] + 2*theta[0]*(theta[1] + theta[0]**2 - 2)
    g2 = theta[1] + theta[0]**2 - 2
    return np.array([g1, g2])

def leapfrog(theta, p, eps, L):
    theta, p = theta.copy(), p.copy()
    p = p - 0.5*eps*grad_U(theta)
    for _ in range(L):
        theta = theta + eps*p  # M = I
        if _ < L - 1:
            p = p - eps*grad_U(theta)
    p = p - 0.5*eps*grad_U(theta)
    return theta, p

def hmc_step(theta, eps, L):
    p = rng.standard_normal(2)
    theta_new, p_new = leapfrog(theta, p, eps, L)
    H_old = U(theta) + 0.5*np.dot(p, p)
    H_new = U(theta_new) + 0.5*np.dot(p_new, p_new)
    if np.log(rng.random()) < H_old - H_new:
        return theta_new, True
    return theta, False

# Run HMC
theta = np.array([0.0, 0.0])
samples = [theta.copy()]
n_accept = 0
eps, L = 0.1, 20
for t in range(5000):
    theta, accepted = hmc_step(theta, eps, L)
    samples.append(theta.copy())
    n_accept += accepted

samples = np.array(samples[500:])  # burn-in
print(f"HMC acceptance: {n_accept/5000:.3f}")

# Compare with random-walk MH
def rw_step(theta, sigma):
    theta_new = theta + sigma*rng.standard_normal(2)
    if np.log(rng.random()) < U(theta) - U(theta_new):
        return theta_new, True
    return theta, False

theta = np.array([0.0, 0.0])
rw_samples = [theta.copy()]
n_acc_rw = 0
for t in range(5000):
    theta, accepted = rw_step(theta, 0.5)
    rw_samples.append(theta.copy())
    n_acc_rw += accepted
rw_samples = np.array(rw_samples[500:])
print(f"RW MH acceptance: {n_acc_rw/5000:.3f}")

# Plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Contour of U
t1 = np.linspace(-3, 3, 200); t2 = np.linspace(-5, 3, 200)
T1, T2 = np.meshgrid(t1, t2)
UU = 0.5*T1**2 + 0.5*(T2 + T1**2 - 2)**2

for ax, samp, title in [(axes[0], samples, 'HMC'),
                         (axes[1], rw_samples, 'Random-Walk MH')]:
    ax.contour(T1, T2, UU, levels=15, colors='gray', alpha=0.4)
    ax.scatter(samp[:, 0], samp[:, 1], s=2, alpha=0.3)
    ax.set_title(f'{title} (n={len(samp)})')
    ax.set_xlim(-3, 3); ax.set_ylim(-5, 3); ax.grid(alpha=0.3)

plt.tight_layout(); plt.savefig('hmc_vs_rw.png', dpi=150); plt.show()
```

**관찰**: HMC가 banana 모양의 **curved posterior를 잘 탐색**, random-walk는 특정 영역에 갇힘.

---

## 🔗 AI/ML 연결

### Stan, PyMC, NumPyro
모두 NUTS(Ch4-04) 기반 — HMC 자동 튜닝 버전.

### BNN HMC
Neal 1996 PhD thesis — 작은 BNN에 HMC 적용 (당대 학술적 기여). 현대엔 **approximate** BNN (Laplace, MC Dropout)이 더 실용.

### SGLD (Ch5-05)
HMC의 stochastic 버전 — mini-batch gradient + Langevin noise.

### Riemannian HMC (RMHMC)
Girolami & Calderhead 2011: $M$을 Fisher로 설정 → geometry-aware HMC. 어려운 posterior에서 우수.

### Symplectic Integrators in Physics
Molecular dynamics에서 동일 수학. HMC는 statistical ML의 physics 차용.

---

## ⚖️ 가정과 한계

| 가정 | 한계 |
|------|------|
| $\nabla U$ 계산 | Discrete $\theta$ 불가 (Gibbs 사용) |
| Differentiable posterior | Non-smooth (ReLU, hinge) 주의 |
| Step size·trajectory length | Hand-tune 어려움 → NUTS |
| Dimension-independent mixing (idealized) | BNN 등 realistic에선 수렴 여전히 느림 |
| Reversible Leapfrog | Other integrators는 acceptance 복잡 |

**실무 팁**:
- Target acceptance $\approx 0.65$ (이론)
- 실전 Stan/PyMC: **0.8** default (보수적) 
- Step size adaptation: dual averaging (NUTS 내장)

---

## 📌 핵심 정리

$$\boxed{H(\theta, p) = U(\theta) + \frac{1}{2}p^T M^{-1} p, \quad U = -\log\pi}$$

$$\boxed{\alpha = \min(1, \exp(H_0 - H_L))}$$

핵심:
- Momentum이 gradient-aware proposal 제공
- Leapfrog: **2차 symplectic** (reversible, volume-preserving)
- Dim-indep mixing (이론적)
- Stan/PyMC 엔진

---

## 🤔 생각해볼 문제

**문제 1** (기초): $L = 1, \epsilon$ small이면 HMC는 무엇과 비슷?

<details>
<summary>해설</summary>

$L = 1$: single Leapfrog step. $\theta' = \theta + \epsilon p, p \sim \mathcal{N}(0, I)$ (approximately) → small Gaussian random walk.

**Langevin dynamics**와 유사: $\theta' = \theta - \frac{\epsilon^2}{2}\nabla U + \epsilon \mathcal{N}(0, I)$ (Euler-Maruyama).

실제로 $L = 1$ HMC = **Metropolis-adjusted Langevin algorithm (MALA)**. Gradient 일부 활용하지만 momentum 효과 없음.

$L$이 커야 HMC의 장점(긴 trajectory) 발휘.

</details>

**문제 2** (심화): HMC에서 mass $M$을 어떻게 선택?

<details>
<summary>해설</summary>

**Optimal $M$**: Fisher 정보 / Hessian (local):
$$M \approx \mathbb{E}[-\nabla^2 \log\pi(\theta)] = F(\theta)$$

이렇게 하면 **모든 방향에서 같은 curvature** → isotropic dynamics → 효율.

**실전**:
- Default: $M = I$
- Adaptive: warmup 중 sample covariance 추정, $M^{-1} = \hat{\text{Cov}}(\theta)$
- **Riemannian HMC**: $M$이 $\theta$ 의존 (fully geometric)

Stan·PyMC: warmup 동안 자동으로 $M^{-1}$ 학습.

</details>

**문제 3** (AI 연결): BNN 전체에 HMC 가능한가?

<details>
<summary>해설</summary>

이론상: 가능. Neal 1996이 보여줌 (작은 NN + 작은 데이터).

**현실적 장애**:
1. **데이터 크기**: Full gradient $\nabla_W L$ at every step = 전체 데이터 pass → mini-batch 필요 → SGLD/SGHMC
2. **파라미터 수**: $10^6$ dim에서 MCMC 수렴 시간 비현실적
3. **Multimodal**: Deep network의 loss landscape multi-mode — HMC 단일 chain이 한 mode만
4. **Stiff dynamics**: Non-smooth activation (ReLU) + data-dependent gradient → $\epsilon$ 매우 작아야

**실전 대안**:
- Laplace (Ch5-02): local Gaussian approx
- MC Dropout (Ch5-04): implicit BNN
- SWAG (Ch5-05): SGD trajectory 활용
- SGLD/SGHMC: 작은 network에 실험적 HMC

현대 Bayesian DL의 dominant paradigm은 **approximate BNN + HMC를 안 씀**. HMC는 중소 Bayesian model (GP, hierarchical regression)에 강력.

</details>

---

<div align="center">

| | | |
|---|---|---|
| [◀ 02. Gibbs Sampler](./02-gibbs-sampler.md) | [📚 README로 돌아가기](../README.md) | [04. No-U-Turn Sampler (NUTS) ▶](./04-nuts.md) |

</div>
