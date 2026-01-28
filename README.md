# Multi-Agent Social Learning Systems – A POMDP Approach

This repository contains the implementation and experimental evaluation of a **POMDP-based privacy-aware social learning framework** designed to mitigate **information cascades** in sequential multi-agent decision-making systems.

The project investigates how **controlled information revelation**, combined with **belief-state reasoning and reinforcement learning**, can significantly improve collective decision accuracy while delaying premature consensus.



##  Project Overview

In classical social learning, agents combine **private observations** with **publicly observed actions** of others. While this can lead to efficient learning, it often results in **information cascades**, where agents ignore their private signals and blindly follow earlier decisions—even if they are wrong.

This project addresses the problem using:
- **Partially Observable Markov Decision Processes (POMDPs)**  
- **Belief-state modeling**
- **Privacy-constrained decision policies**
- **Reinforcement Learning (Q-learning and SARSA)**

The core idea is simple yet powerful:
> *Neither always revealing nor always hiding information is optimal — instead, agents should adaptively decide **when** to reveal or hide information based on current belief uncertainty.*



##  Objectives

- Model sequential social learning using **MDP and POMDP frameworks**
- Analyze how **information cascades** emerge
- Design a **belief-dependent privacy policy** (Reveal / Hide)
- Learn optimal policies using:
  - **Dynamic Programming (POMDP)**
  - **Q-Learning (off-policy RL)**
  - **SARSA (on-policy RL)**
- Compare performance against baseline strategies



##  Methodology

### 1. Decision Frameworks
| Model | Observability | Learning | Social Interaction |
|------|--------------|----------|-------------------|
| MDP | Fully observable | Model-based |
| POMDP | Partially observable | Model-based | 
| POMDP-SL | Partial + public belief | Model-based | 
| RL-SL | Partial + public belief | Model-free | 

### 2. Social Learning Model
- Hidden binary state `x ∈ {0,1}`
- Sequential agents receive noisy private signals
- Public belief updated via Bayesian inference
- Agents choose between:
  - **REVEAL** → share raw signal
  - **HIDE** → share only final decision

### 3. POMDP-Based Privacy Policy
- Belief space discretized over `[0,1]`
- Optimal policy computed via **backward dynamic programming**
- Results in a **threshold-based policy**:
  - Reveal when belief uncertainty is high
  - Hide when belief is strong (near 0 or 1)



##  Reinforcement Learning Approaches

To scale beyond model-based solutions, the belief-MDP is solved using:

###  Q-Learning
- Off-policy TD control
- Faster convergence
- Learns sharp belief thresholds

###  SARSA
- On-policy TD control
- More conservative, smoother learning
- Slightly higher threshold due to exploration effects

Both methods:
- Operate over discretized belief states
- Use ε-greedy exploration with decay
- Converge to stable, interpretable policies



##  Experimental Results

### Compared Strategies
1. **Always Reveal**
2. **Always Hide** (classical social learning)
3. **Optimal POMDP Policy**
4. **Learned RL Policies (Q-learning, SARSA)**

### Key Findings
- Always Hide → **early cascades**, poor accuracy
- Always Reveal → **noisy belief updates**
- **POMDP & RL policies**:
  - Delay cascade formation
  - Preserve belief diversity
  - Achieve **higher collective accuracy**
  - Learn monotonic **threshold-based policies**



##  Simulation Setup

- Number of agents: `N = 20`
- Signal accuracy: `psignal ∈ {0.55, 0.65, 0.75, 0.90}`
- Belief grid: `101` points
- Evaluation via Monte Carlo simulations
- Metrics:
  - Public belief evolution
  - Average decision accuracy
  - RL convergence diagnostics

---

## 📁 Repository Structure (suggested)

