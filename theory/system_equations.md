# Core System Equations

## 1. System State Evolution (Master Equation)

The deployed AI is modeled as a discrete-time dynamical system:

$$S_{t+1} = F(S_t, D_t, F_t, C_t, R_t)$$

**Interpretation:** The system state at time $t+1$ depends on:
- Current state $S_t$ (inertia, history)
- Drift signal $D_t$ (environmental pressure)
- Feedback signal $F_t$ (learning signal)
- Cost constraints $C_t$ (computational/operational limits)
- Risk bounds $R_t$ (safety/compliance)

---

## 2. System State Vector

The system state is *not* just a model; it is:

$$S_t = \{θ_t, W_t, \mathbf{σ}_t^{SHAP}, \mathbf{E}_t, π_t\}$$

**Components:**

| Component | Symbol | Meaning |
|-----------|--------|---------|
| Model parameters | $θ_t$ | Neural network / tree weights |
| Data window | $W_t$ | Recent training examples |
| Feature importance | $\mathbf{σ}_t^{SHAP}$ | SHAP values per feature |
| Error metrics | $\mathbf{E}_t$ | {accuracy, precision, recall, AUC} |
| Policy state | $π_t$ | Retraining threshold, feedback mode |

---

## 3. Drift Signal Vector

Drift is *not* a binary decision; it is a multi-dimensional signal:

$$D_t = [D_t^{KS}, D_t^{ADWIN}, D_t^{Error}, D_t^{SHAP}]$$

**Components:**

### 3.1 Kolmogorov-Smirnov (Covariate Drift)
$$D_t^{KS} = \max_x |P_t(x) - P_{t+\tau}(x)|$$
Measures distribution shift in input features.

### 3.2 ADWIN (Adaptive Windowing)
$$D_t^{ADWIN} = \begin{cases} 1 & \text{if } \text{error change exceeds } \delta \\ 0 & \text{otherwise} \end{cases}$$
Detects concept drift via adaptive windowing.

### 3.3 Error-Based Drift
$$D_t^{Error} = E_t - E_{t-1}$$
Raw performance degradation.

### 3.4 Attribution Drift (Novel)
$$D_t^{SHAP} = \left\| \mathbf{σ}_t^{SHAP} - \mathbf{σ}_{t-1}^{SHAP} \right\|_2$$
Measures changes in feature importance rankings.

---

## 4. Feedback Signal

Feedback is modeled as a probabilistic signal:

$$F_t \sim P(F_t | trust, cost) = \frac{\exp(\beta_1 \cdot trust - \beta_2 \cdot cost)}{Z}$$

**Parameters:**
- $trust$: Human confidence in model (0 to 1)
- $cost$: Labeling cost per sample (time, money, etc.)

**Regimes:**

### 4.1 Passive Feedback
$$F_t = \mathbb{1}[\text{error detected}]$$
No active human intervention.

### 4.2 Human-in-the-Loop
$$F_t \sim \text{Bernoulli}(p_t)$$
where $p_t$ depends on uncertainty sampling.

### 4.3 Policy Feedback
$$F_t = \mathbb{1}[\text{safety rule violated}]$$
Business rules override model decisions.

---

## 5. Cost Function

Total operational cost:

$$C_t = \underbrace{c_{\text{compute}} \cdot \mathbb{1}[\text{retrain}]}_{\text{compute}} + \underbrace{c_{\text{label}} \cdot |F_t|}_{\text{labeling}} + \underbrace{c_{\text{deploy}}}_{\text{deployment}}$$

**Constraints:**
$$C_t \leq C_{\text{budget}}$$

---

## 6. Risk Function

Multi-dimensional risk:

$$R_t = \begin{pmatrix} R_t^{\text{accuracy}} \\ R_t^{\text{safety}} \\ R_t^{\text{fairness}} \\ R_t^{\text{stability}} \end{pmatrix}$$

### 6.1 Accuracy Risk
$$R_t^{\text{accuracy}} = 1 - E_t$$

### 6.2 Safety Risk (e.g., critical systems)
$$R_t^{\text{safety}} = \mathbb{1}[\text{high error on critical class}]$$

### 6.3 Fairness Risk
$$R_t^{\text{fairness}} = \max_g \left| E_t(g) - \bar{E}_t \right|$$
where $g$ is a demographic group.

### 6.4 Stability Risk
$$R_t^{\text{stability}} = \left\| θ_t - θ_{t-1} \right\|_2$$
Change in model parameters.

**Constraints:**
$$R_t \leq R_{\text{max}}$$

---

## 7. Retraining Decision

The system decides whether to retrain based on:

$$\text{retrain}_t = \arg\max_{\text{yes/no}} \left[ \mathcal{L}(θ_t | S_{t+1}) - \mathcal{L}(θ_{t-1} | S_{t+1}) \right]$$

subject to:
- Cost: $C_t \leq C_{\text{budget}}$
- Risk: $R_t \leq R_{\text{max}}$

---

## 8. Loss Function (Multi-Objective)

The overall loss surface balancing accuracy, cost, and risk:

$$L_t = α \cdot E_t + β \cdot C_t + γ \cdot \|R_t\|_1$$

where $α, β, γ > 0$ are weighted priorities.

**Interpretation:**
- High $α$: Accuracy-critical systems (e.g., medical diagnosis)
- High $β$: Cost-sensitive systems (e.g., edge devices)
- High $γ$: Risk-sensitive systems (e.g., autonomous vehicles)

---

## 9. Stability-Plasticity Index (SPI)

Quantifies the trade-off between adaptation and stability:

$$SPI_t = \frac{\Delta E_t}{\left\| θ_t - θ_{t-1} \right\|_2 + \epsilon}$$

where $\epsilon$ is a small regularizer to avoid division by zero.

**Interpretation:**
- **SPI > 1**: Large accuracy gains with modest parameter changes (good plasticity)
- **SPI ≈ 0.1**: Small accuracy gains despite large parameter changes (over-retraining)
- **SPI → 0**: No improvement despite retraining (instability / overfitting)

---

## 10. Time Evolution Summary

**In discrete time steps:**

1. **Observe** drift signal: $D_t = [D_t^{KS}, D_t^{ADWIN}, D_t^{Error}, D_t^{SHAP}]$
2. **Receive** feedback: $F_t \sim P(\cdot | trust, cost)$
3. **Evaluate** current risk: $R_t$
4. **Check** cost budget: $C_t \leq C_{\text{budget}}$
5. **Decide** retraining: $\text{retrain}_t = f(D_t, F_t, R_t, C_t)$
6. **Update** state: $S_{t+1} = F(S_t, D_t, F_t, C_t, R_t)$
7. **Compute** SPI: $SPI_t$ for analysis

---

## References

- **Stability-Plasticity Dilemma**: Grossberg (1980), Mermillod et al. (2013)
- **Concept Drift**: Gama et al. (2014)
- **Cost-Sensitive Learning**: Elkan (2001)
- **Fairness Constraints**: Moritz et al. (2020)

