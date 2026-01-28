import numpy as np
import random


# Parameters

p_signal = 0.75
pi0 = 0.5
HORIZON = 30
belief_grid = np.linspace(0, 1, 101)

# Learning parameters
alpha0 = 0.1
gamma = 0.95

epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.9997   # slower decay


MAX_EPISODES = 120000
CONV_THRESHOLD = 5e-3    # more realistic for noisy POMDP
CHECK_INTERVAL = 2000
REQUIRED_STABLE = 3

EPS = 1e-12


# Helper functions

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


# Q-Learning


Q = np.zeros((len(belief_grid), 2))

def choose_action(s, eps):
    if random.random() < eps:
        return random.randint(0,1)
    return np.argmax(Q[s])

print("\nTraining started...\n")

Q_snap = Q.copy()
stable_count = 0
global_step = 0

for ep in range(MAX_EPISODES):

    # Sample true state x and initial belief b
    x = 1 if random.random() < pi0 else 0
    b = random.choice(list(belief_grid))   # random starting belief
    s = belief_to_index(b)

    for t in range(HORIZON):

        a = choose_action(s, epsilon)

        # Sample private signal y given true state x
        y = 1 if random.random() < (p_signal if x==1 else 1-p_signal) else 0

        if a == 0:      # REVEAL
            public_a = y
            b2 = update_belief_with_reveal(b, y)
        else:           # HIDE
            public_a = myopic_action(b, y)
            b2 = update_belief_with_action(b, public_a)

        # Reward: 1 if public action matches true state
        r = 1 if public_a == x else 0
        r = float(r)

        s2 = belief_to_index(b2)

        # ---- Decaying learning rate (critical for convergence) ----
        global_step += 1
        # Faster decay than before:

        alpha_t = alpha0 / (1 + 0.0002 * global_step)


        # Q-learning update
        Q[s,a] += alpha_t * (r + gamma*np.max(Q[s2]) - Q[s,a])

        b, s = b2, s2

    # decay epsilon (goes to 0 → greedy in the limit)
        epsilon = max(0.05, epsilon * 0.9997)

    # ---- Convergence check every 1000 episodes ----
    if ep % CHECK_INTERVAL == 0 and ep > 0:
        diff = np.max(np.abs(Q - Q_snap))
        print(f"Episode {ep:6d} | max ΔQ = {diff:.6f} | ε = {epsilon:.4f}")

        if diff < CONV_THRESHOLD:
            stable_count += 1
            if stable_count >= REQUIRED_STABLE:
                print(f"\n Converged after {ep} episodes.")
                break
        else:
            stable_count = 0

        Q_snap = Q.copy()

print("\nTraining Complete.")

# ----------------------------
# VALUE FUNCTION + POLICY
# ----------------------------
V = np.max(Q, axis=1)
policy = np.argmax(Q, axis=1)

np.set_printoptions(precision=4, suppress=True)

print("\nFinal Q(S,A):\n")
print(Q)

print("\nFinal V(S) = max_a Q(S,a):\n")
print(V)

print("\nLearned Q Policy (0 = REVEAL, 1 = HIDE):\n")
print(policy)

print("\nPolicy intervals:")
cur = policy[0]
start = 0

for i, val in enumerate(policy):
    if val != cur:
        print(f"Action {cur} for belief [{belief_grid[start]:.3f},{belief_grid[i-1]:.3f}]")
        cur = val
        start = i
print(f"Action {cur} for belief [{belief_grid[start]:.3f},{belief_grid[-1]:.3f}]")
