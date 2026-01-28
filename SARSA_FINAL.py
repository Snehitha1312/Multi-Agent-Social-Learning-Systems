import numpy as np
import random


# Parameters

p_signal = 0.75
pi0 = 0.5
HORIZON = 20
belief_grid = np.linspace(0, 1, 101)

gamma = 0.95

# Learning + exploration
alpha0 = 0.1               # initial learning rate
epsilon = 1.0              # start with more exploration
epsilon_min = 0.0          # eventually go fully greedy
epsilon_decay = 0.9995

MAX_EPISODES = 40000
CONV_THRESHOLD = 5e-3      # practical convergence threshold
CHECK_INTERVAL = 1000      # check every 1000 episodes
REQUIRED_STABLE = 3        # need 3 consecutive small ΔQ checks

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
        if myopic_action(prior, y) == a:
            s += P_y_given_x(y, x)
    return s

def update_belief_with_action(prior, a):
    la1 = likelihood_of_action(a, prior, 1)
    la0 = likelihood_of_action(a, prior, 0)
    num = la1 * prior
    den = num + la0 * (1-prior)
    return prior if den < EPS else num/den





Q = np.zeros((len(belief_grid), 2))   # 0=REVEAL, 1=HIDE

def choose_action(s, eps):
    if random.random() < eps:
        return random.randint(0, 1)
    return int(np.argmax(Q[s]))

print("\nSARSA training started...\n")

global_step = 0
Q_snap = Q.copy()
stable_count = 0

for ep in range(MAX_EPISODES):

    # Sample hidden state
    x = 1 if random.random() < pi0 else 0

    # Start from a random belief (better state coverage)
    b = random.choice(belief_grid)
    s = belief_to_index(b)
    a = choose_action(s, epsilon)

    for t in range(HORIZON):

        # Private signal
        y = 1 if random.random() < (p_signal if x == 1 else 1 - p_signal) else 0

        # Action semantics: REVEAL vs HIDE
        if a == 0:  # REVEAL
            public_a = y
            b2 = update_belief_with_reveal(b, y)
        else:       # HIDE
            public_a = myopic_action(b, y)
            b2 = update_belief_with_action(b, public_a)

        # Reward: correct guess of x?
        r = 1 if public_a == x else 0

        # Next state + next action (SARSA)
        s2 = belief_to_index(b2)
        a2 = choose_action(s2, epsilon)

        # Decaying learning rate
        global_step += 1
        alpha_t = alpha0 / (1 + 1e-5 * global_step)

        # SARSA update
        Q[s, a] += alpha_t * (r + gamma * Q[s2, a2] - Q[s, a])

        # Move on
        b, s, a = b2, s2, a2

    # Epsilon decay per episode
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Convergence check
    if ep % CHECK_INTERVAL == 0 and ep > 0:
        diff = np.max(np.abs(Q - Q_snap))
        print(f"Episode {ep:5d} | max ΔQ = {diff:.6f} | ε = {epsilon:.4f}")

        if diff < CONV_THRESHOLD:
            stable_count += 1
            if stable_count >= REQUIRED_STABLE:
                print(f"\n SARSA converged after {ep} episodes.")
                break
        else:
            stable_count = 0

        Q_snap = Q.copy()

print("\nSARSA training complete.")


# VALUE FUNCTION + POLICY

V = np.max(Q, axis=1)
policy = np.argmax(Q, axis=1)

np.set_printoptions(precision=4, suppress=True)

print("\nFinal Q(S,A):\n")
print(Q)

print("\nFinal V(S) = max_a Q(S,a):\n")
print(V)

print("\nLearned SARSA policy (0=REVEAL, 1=HIDE):")
print(policy)

print("\nPolicy intervals:")
cur = policy[0]
start = 0
for i, val in enumerate(policy):
    if val != cur:
        print(f"Action {cur} on belief [{belief_grid[start]:.3f}, {belief_grid[i-1]:.3f}]")
        cur = val
        start = i
print(f"Action {cur} on belief [{belief_grid[start]:.3f}, {belief_grid[-1]:.3f}]")
