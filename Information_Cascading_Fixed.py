"""
POMDP-based Privacy Policy to Delay Information Cascades
---------------------------------------------------------
Implements:
1. MDP-like binary-state model with partial observations.
2. Sequential agents with two privacy choices:
   - REVEAL: broadcast private observation (y)
   - HIDE: broadcast MAP decision (a)
3. Finite-horizon POMDP solved by Dynamic Programming over belief grid.
4. Compares Always Reveal, Always Hide, and Optimal POMDP policies.
"""

import numpy as np
import random
import matplotlib.pyplot as plt

# --------------------------------------------------
# Global Parameters
# --------------------------------------------------
p_signal = 0.75   # private observation accuracy
pi0 = 0.5         # initial prior P(x=1)
HORIZON = 20      # number of sequential agents
belief_grid = np.linspace(0.0, 1.0, 101)
EPS = 1e-12       # epsilon for numerical stability

# --------------------------------------------------
# Helper Functions
# --------------------------------------------------
def belief_to_index(b, grid):
    """Return index of nearest belief grid point."""
    b = float(np.clip(b, 0.0, 1.0))
    return int(np.abs(grid - b).argmin())

def P_y_given_x(y, x):
    """Observation likelihood."""
    return p_signal if y == x else (1 - p_signal)

def update_belief_with_reveal(prior, y):
    """Bayesian update when raw observation y is revealed."""
    num = P_y_given_x(y, 1) * prior
    den = num + P_y_given_x(y, 0) * (1 - prior)
    if den < EPS:
        return prior
    return num / den

def myopic_action_from_private(prior, y):
    """MAP decision (a) based on private observation y and prior."""
    lik1 = P_y_given_x(y, 1)
    lik0 = P_y_given_x(y, 0)
    num = lik1 * prior
    den = num + lik0 * (1 - prior)
    post = num / den if den > EPS else 0.5
    return 1 if post >= 0.5 else 0

def likelihood_of_action_given_state(a, prior, x):
    """Compute P(a | x, prior) by marginalizing over y."""
    s = 0.0
    for y in (0, 1):
        if myopic_action_from_private(prior, y) == a:
            s += P_y_given_x(y, x)
    return s

def update_belief_with_action(prior, a):
    """Bayesian update when only the MAP action is observed."""
    la1 = likelihood_of_action_given_state(a, prior, 1)
    la0 = likelihood_of_action_given_state(a, prior, 0)
    num = la1 * prior
    den = num + la0 * (1 - prior)
    if den < EPS:
        return prior
    return num / den

# --------------------------------------------------
# Dynamic Programming for Optimal Policy
# --------------------------------------------------
def compute_optimal_policy():
    n = len(belief_grid)
    V = np.zeros((HORIZON + 1, n))
    POLICY = np.zeros((HORIZON, n), dtype=int)  # 0=REVEAL, 1=HIDE

    trans_reveal, trans_hide = {}, {}
    for i, b in enumerate(belief_grid):
        # --- REVEAL transition ---
        immediate_reveal = p_signal  # expected correctness = signal accuracy
        b_y0 = update_belief_with_reveal(b, 0)
        b_y1 = update_belief_with_reveal(b, 1)
        py0 = (1 - b) * P_y_given_x(0, 0) + b * P_y_given_x(0, 1)
        py1 = 1 - py0
        trans_reveal[i] = (immediate_reveal, [(py0, b_y0), (py1, b_y1)])

        # --- HIDE transition ---
        immediate_hide = 0.0
        pa_and_next = {}
        for y in (0, 1):
            a = myopic_action_from_private(b, y)
            for x in (0, 1):
                prob_x = b if x == 1 else (1 - b)
                immediate_hide += prob_x * P_y_given_x(y, x) * (1 if a == x else 0)
            pa_and_next[a] = pa_and_next.get(a, 0.0) + ((1 - b) * P_y_given_x(y, 0) + b * P_y_given_x(y, 1))

        transitions = [(pa, update_belief_with_action(b, a)) for a, pa in pa_and_next.items()]
        assert abs(sum(pa for pa, _ in transitions) - 1.0) < 1e-6
        trans_hide[i] = (immediate_hide, transitions)

    # --- Backward DP ---
    for t in range(HORIZON - 1, -1, -1):
        for i, b in enumerate(belief_grid):
            # Evaluate both choices
            val_r = trans_reveal[i][0] + sum(prob * V[t + 1, belief_to_index(nb, belief_grid)] for prob, nb in trans_reveal[i][1])
            val_h = trans_hide[i][0] + sum(prob * V[t + 1, belief_to_index(nb, belief_grid)] for prob, nb in trans_hide[i][1])
            if val_r >= val_h:
                V[t, i] = val_r
                POLICY[t, i] = 0
            else:
                V[t, i] = val_h
                POLICY[t, i] = 1
    return V, POLICY

# --------------------------------------------------
# Simulation Utilities
# --------------------------------------------------
def simulate_run(strategy_func, POLICY=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    x = 1 if random.random() < pi0 else 0  # true state
    b = pi0
    beliefs, broadcasts, private_obs, actions, correctness = [b], [], [], [], []
    for t in range(HORIZON):
        y = 1 if random.random() < (p_signal if x == 1 else (1 - p_signal)) else 0
        private_obs.append(y)
        decision = strategy_func(t, b, POLICY)
        broadcasts.append(decision)
        if decision == 0:  # reveal
            a = y
            correctness.append(1 if a == x else 0)
            b = update_belief_with_reveal(b, a)
        else:
            a = myopic_action_from_private(b, y)
            correctness.append(1 if a == x else 0)
            b = update_belief_with_action(b, a)
        beliefs.append(b)
        actions.append(a)
    return dict(true_state=x, beliefs=beliefs, broadcasts=broadcasts,
                private_obs=private_obs, actions=actions, correctness=correctness)

def always_reveal_strategy(t, b, POLICY=None): return 0
def always_hide_strategy(t, b, POLICY=None): return 1
def optimal_policy_strategy(t, b, POLICY):
    return int(POLICY[t, belief_to_index(b, belief_grid)])

def eval_strategy(strategy, POLICY=None, runs=200):
    total = []
    for s in range(runs):
        r = simulate_run(strategy, POLICY, seed=s)
        total.append(sum(r["correctness"]))
    return np.mean(total) / HORIZON

# --------------------------------------------------
# Main Execution
# --------------------------------------------------
if __name__ == "__main__":
    V, POLICY = compute_optimal_policy()

    avg_reveal = eval_strategy(always_reveal_strategy)
    avg_hide = eval_strategy(always_hide_strategy)
    avg_opt = eval_strategy(optimal_policy_strategy, POLICY)
    print(f"\nAverage fraction correct per agent:")
    print(f"Always Reveal : {avg_reveal:.3f}")
    print(f"Always Hide   : {avg_hide:.3f}")
    print(f"Optimal POMDP : {avg_opt:.3f}")

    # Representative HIDE-run (cascade example)
    rep = simulate_run(always_hide_strategy, seed=7)
    print("\nRepresentative HIDE-run (showing cascade example):")
    print("True state:", "Red(1)" if rep["true_state"] else "Blue(0)")
    for t in range(8):
        print(f"t={t:2d} prior={rep['beliefs'][t]:.3f} private_obs={rep['private_obs'][t]} "
              f"broadcast={'HIDE' if rep['broadcasts'][t] else 'REVEAL'} "
              f"action={rep['actions'][t]} correct={rep['correctness'][t]} next_prior={rep['beliefs'][t+1]:.3f}")

    # Cascade example
    for t in range(HORIZON):
        prior, y, a = rep['beliefs'][t], rep['private_obs'][t], rep['actions'][t]
        favored = 1 if prior >= 0.5 else 0
        if y != favored and a == favored:
            print(f"\n*** Cascade example found at t={t}: prior={prior:.3f} favors {favored}, private_obs={y}, action={a}")
            break

    # Plot belief evolution
    plt.figure(figsize=(9,4))
    for strategy, label in [(always_reveal_strategy,"Always Reveal"),
                            (always_hide_strategy,"Always Hide"),
                            (optimal_policy_strategy,"Optimal POMDP")]:
        run = simulate_run(strategy, POLICY, seed=2)
        plt.plot(run['beliefs'], label=label)
    plt.xlabel("Agent index (t)")
    plt.ylabel("Public belief P(x=1)")
    plt.title("Public belief evolution under different strategies")
    plt.legend(); plt.grid(True)
    plt.tight_layout()
    plt.savefig("belief_evolution.png", dpi=300)
    plt.show()


    # Show optimal policy intervals at t=0
    print("\nOptimal policy at t=0 (0=REVEAL, 1=HIDE) intervals:")
    policy_at_t0 = POLICY[0, :]
    intervals, cur, start = [], policy_at_t0[0], 0
    for i, val in enumerate(policy_at_t0):
        if val != cur:
            intervals.append((cur, belief_grid[start], belief_grid[i-1]))
            start, cur = i, val
    intervals.append((cur, belief_grid[start], belief_grid[-1]))
    for val, a, b in intervals:
        print(f"Policy {val} on belief in [{a:.3f}, {b:.3f}]")
