# qlearning_convergence_noheatmap.py
import numpy as np
import random
import matplotlib.pyplot as plt
import os

# ----------------------------
# Parameters
# ----------------------------
p_signal = 0.75
pi0 = 0.5
HORIZON = 30
belief_grid = np.linspace(0, 1, 101)

alpha0 = 0.1
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.0
epsilon_decay = 0.9995

MAX_EPISODES = 120000
CONV_THRESHOLD = 5e-3
CHECK_INTERVAL = 1000
REQUIRED_STABLE = 3

EPS = 1e-12

# ----------------------------
# Helper functions
# ----------------------------
def belief_to_index(b):
    return int(np.abs(belief_grid - b).argmin())

def P_y_given_x(y, x):
    return p_signal if y == x else (1 - p_signal)

def update_belief_with_reveal(prior, y):
    num = P_y_given_x(y,1) * prior
    den = num + P_y_given_x(y,0) * (1-prior)
    return prior if den < EPS else num/den

def myopic_action(prior, y):
    lik1 = P_y_given_x(y,1)
    lik0 = P_y_given_x(y,0)
    num = lik1 * prior
    den = num + lik0 * (1-prior)
    post = num/den if den > EPS else 0.5
    return 1 if post >= 0.5 else 0

def likelihood_of_action(a, prior, x):
    s = 0
    for y in (0,1):
        if myopic_action(prior,y) == a:
            s += P_y_given_x(y,x)
    return s

def update_belief_with_action(prior, a):
    la1 = likelihood_of_action(a, prior, 1)
    la0 = likelihood_of_action(a, prior, 0)
    num = la1 * prior
    den = num + la0 * (1-prior)
    return prior if den < EPS else num/den

# ----------------------------
# Q-Learning
# ----------------------------
Q = np.zeros((len(belief_grid), 2))

def choose_action(s, eps):
    if random.random() < eps:
        return random.randint(0,1)
    return int(np.argmax(Q[s]))

print("\nQ-Learning training started...\n")

Q_snap = Q.copy()
stable_count = 0
global_step = 0

checkpoints = []
max_deltas = []
V_history = []
rep_indices = [0, 25, 50, 75, 100]  # representative beliefs to plot

for ep in range(1, MAX_EPISODES+1):
    x = 1 if random.random() < pi0 else 0
    b = random.choice(list(belief_grid))
    s = belief_to_index(b)

    for t in range(HORIZON):
        a = choose_action(s, epsilon)
        y = 1 if random.random() < (p_signal if x==1 else 1-p_signal) else 0

        if a == 0:
            public_a = y
            b2 = update_belief_with_reveal(b, y)
        else:
            public_a = myopic_action(b, y)
            b2 = update_belief_with_action(b, public_a)

        r = 1 if public_a == x else 0
        s2 = belief_to_index(b2)

        global_step += 1
        alpha_t = alpha0 / np.sqrt(max(1, global_step))
        Q[s,a] += alpha_t * (r + gamma*np.max(Q[s2]) - Q[s,a])

        b, s = b2, s2

    # decay epsilon
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    if ep % CHECK_INTERVAL == 0:
        diff = np.max(np.abs(Q - Q_snap))
        print(f"Episode {ep:6d} | max ΔQ = {diff:.6f} | ε = {epsilon:.4f}")

        checkpoints.append(ep)
        max_deltas.append(diff)
        V_history.append(np.max(Q, axis=1).copy())

        if diff < CONV_THRESHOLD:
            stable_count += 1
            if stable_count >= REQUIRED_STABLE:
                print(f"\nConverged after {ep} episodes.")
                break
        else:
            stable_count = 0

        Q_snap = Q.copy()

print("\nQ-Learning training complete.\n")

os.makedirs("images", exist_ok=True)

# Plot: max ΔQ vs episodes
plt.figure(figsize=(8,4))
plt.plot(checkpoints, max_deltas, marker='o', linewidth=1)
plt.xlabel('Episodes')
plt.ylabel('max ΔQ (checkpoint)')
plt.title('Q-learning: max ΔQ vs Episodes')
plt.grid(True)
plt.tight_layout()
plt.savefig("images/qlearning_maxdelta.png", dpi=200)

# Plot: V(b) convergence for representative beliefs
plt.figure(figsize=(8,4))
for idx in rep_indices:
    vals = [V[idx] for V in V_history]
    plt.plot(checkpoints, vals, marker='o', label=f'b={belief_grid[idx]:.2f}')
plt.xlabel('Episodes')
plt.ylabel('V(b) (checkpoint)')
plt.title('Q-learning: V(b) convergence (representative beliefs)')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.savefig("images/qlearning_V_convergence.png", dpi=200)

print("Saved q-learning plots in images/")
