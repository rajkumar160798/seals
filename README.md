# SEALS: Self-Evolving AI Lifecycle Simulator

A reference implementation for self-evolving machine learning systems. This is not just code — it is a scientific instrument to study deployed AI behavior.

## What This Project Is (High Level)

SEALS is a controlled experimental environment that simulates:

- A deployed ML system operating under non-stationary data
- Receiving feedback (human + automated)
- Making retraining decisions
- Balancing stability vs plasticity under explicit cost and risk constraints

This project exists to validate equations, not to win Kaggle.

## Core System Abstraction

The deployed AI is modeled as a dynamical system:

$$S_{t+1} = F(S_t, D_t, F_t, C_t, R_t)$$

**Where:**

| Symbol | Meaning |
|--------|---------|
| $S_t$ | System state (model, data window, explanations, policy) |
| $D_t$ | Drift signal |
| $F_t$ | Feedback signal |
| $C_t$ | Computational / operational cost |
| $R_t$ | Risk (error, safety, compliance) |

## Core Modules

### 1. Drift Engine (`simulator/drift_engine.py`)
Simulates controlled drift with multiple mechanisms:
- Covariate drift
- Label drift
- Concept drift
- Feedback-induced drift

### 2. System State Tracker
Tracks complete ML system evolution:
```python
SystemState = {
    "model_weights": θ_t,
    "data_window": W_t,
    "feature_importance": SHAP_t,
    "error_metrics": E_t,
    "policy_state": π_t
}
```

### 3. Drift Signal Generator (`simulator/drift_engine.py`)
Unifies multiple drift signals into a vector:

$$D_t = [D_t^{KS}, D_t^{ADWIN}, D_t^{Error}, D_t^{SHAP}]$$

### 4. Feedback Channel (`simulator/feedback_engine.py`)
Implements three feedback regimes:
- **Passive feedback**: Error-based, no human intervention
- **Human-in-the-loop**: Selective relabeling / override
- **Policy feedback**: Business rule / safety intervention

### 5. Retraining Policy (`simulator/retraining_policy.py`)
Defines retraining regimes with control theory:

| Regime | Behavior |
|--------|----------|
| Over-plastic | Retrains too often (unstable) |
| Over-stable | Retrains too rarely (stale) |
| Balanced | Optimal evolution |

### 6. Cost–Risk Surface (`metrics/spi.py`)
Explicitly models trade-offs:

$$L = \alpha \cdot Error + \beta \cdot Cost + \gamma \cdot Risk$$

**Stability-Plasticity Index (SPI):**

$$SPI = \frac{\Delta Accuracy}{\Delta Model Change}$$

## Experiments

### Experiment 1: Stability vs Plasticity
- Shows oscillation with excessive retraining
- Shows collapse with insufficient retraining
- Plots: Accuracy vs Time, SPI vs Time

### Experiment 2: Feedback Regimes
- Compares: No feedback, sparse feedback, dense feedback
- Demonstrates non-linear returns to human feedback

### Experiment 3: Drift + Explanation Failure
- Cases where accuracy stays high but SHAP explanations drift
- Foreshadows attribution drift analysis

## Project Structure

```
seals/
├── theory/
│   ├── system_equations.md
│   └── stability_plasticity.md
├── simulator/
│   ├── __init__.py
│   ├── drift_engine.py
│   ├── feedback_engine.py
│   └── retraining_policy.py
├── experiments/
│   ├── __init__.py
│   ├── exp_stability_plasticity.py
│   └── exp_feedback_regimes.py
├── metrics/
│   ├── __init__.py
│   ├── spi.py
│   └── attribution_drift.py
├── paper/
│   ├── main.tex
│   └── figures/
├── data/
├── notebooks/
└── README.md
```

## Key Concepts

### System State Evolution
Rather than tracking "a model," SEALS tracks the entire system state including:
- Model parameters and performance
- Data window characteristics
- Feature importance (SHAP)
- Error metrics
- Policy decisions

### Drift Signal Vector
Drift is not binary; it's a multi-dimensional signal combining:
- **KS test**: Covariate distribution changes
- **ADWIN**: Adaptive windowing drift
- **Error-based**: Performance degradation
- **SHAP-based**: Attribution drift (Paper 2)

### Feedback as Control Signal
Feedback is modeled probabilistically, capturing:
- Cost of obtaining labels
- Human trust in the system
- Effectiveness of corrections

### Stability-Plasticity Index
The SPI metric quantifies the trade-off between:
- **Plasticity**: Ability to adapt to new data
- **Stability**: Resistance to destructive changes

## What This Is NOT

❌ Not a SaaS  
❌ Not a dashboard-heavy demo  
❌ Not model benchmarking  

It is a scientific simulator — like an ML wind tunnel.

## Getting Started

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or install as development package
pip install -e .
```

### Running Experiments

Run all experiments at once:
```bash
python run_all_experiments.py
```

Or run individual experiments:
```bash
# Experiment 1: Stability vs Plasticity (synthetic data)
python experiments/exp_stability_plasticity.py

# Experiment 2: Feedback Regimes (synthetic data)
python experiments/exp_feedback_regimes.py

# Experiment 3: Real Dataset Evaluation (CMAPSS + AI4I)
python experiments/exp_real_datasets.py
```

### Datasets

The following datasets are included in `data/`:

- **CMAPSS** (`data/CMAPSSData/`): NASA Commercial Modular Aero-Propulsion System Simulation
  - Multiple engine degradation datasets (FD001, FD002, FD003, FD004)
  - Time-series engine sensor readings
  - Remaining Useful Life (RUL) labels
  - Used for: Predictive maintenance under concept drift
  - Reference: [NASA Dataset](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/)

- **AI4I 2020** (`data/ai4i2020.csv`): Predictive Maintenance Dataset
  - Industrial equipment sensor data
  - Machine failure classification
  - 14 features including temperature, humidity, power
  - Used for: Binary classification with imbalanced data
  - Reference: [UCI ML Repository](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)

### Example Usage

```python
from simulator import get_cmapss_data, get_ai4i_data

# Load CMAPSS
cmapss_data = get_cmapss_data(dataset="FD001", normalize=True)
X_train, y_train = cmapss_data["X_train"], cmapss_data["y_train"]

# Load AI4I
ai4i_data = get_ai4i_data(normalize=True)
X_train, y_train = ai4i_data["X_train"], ai4i_data["y_train"]
features = ai4i_data["feature_names"]
```

## Theory References

- See `theory/system_equations.md` for mathematical foundations
- See `theory/stability_plasticity.md` for control theory basis

## Contributing

This is a research project. Submit issues for bugs or theoretical concerns.

## License

MIT
