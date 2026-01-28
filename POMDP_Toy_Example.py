import numpy as np

# Parameters
p = 0.9   # probability machine stays good
q = 0.5   # probability new machine is poor
c1 = 1.0  # cost when machine is good
c2 = 3.0  # cost when machine is poor
R  = 5.0  # replacement cost
alpha = 0.95  # discount factor
N = 10  # horizon length

# Belief discretization (pi2 = prob machine is poor)
grid = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]

# Initialize value function
J = np.zeros((N+1, len(grid)))  # J[k, i] = value at step k with pi2 = grid[i]

# Helper: expected immediate costs
def cost(pi2, action):
    pi1 = 1 - pi2
    if action == 0:  # keep machine
        return pi1 * c1 + pi2 * c2
    else:  # replace machine
        return R

# Belief update (Bayes filter)
def update_belief(pi2, action):
    # For toy example, we use transition matrices
    if action == 0:  # keep
        pi2_next = pi2 + (1 - pi2) * (1 - p)  # chance of degrading
    else:  # replace
        pi2_next = q  # replacement resets belief
    return pi2_next

# Value iteration backward
for k in range(N-1, -1, -1):
    for i, pi2 in enumerate(grid):
        # action 0: keep
        pi2_next = update_belief(pi2, 0)
        J_keep = cost(pi2, 0) + alpha * np.interp(pi2_next, grid, J[k+1])

        # action 1: replace
        pi2_next = update_belief(pi2, 1)
        J_replace = cost(pi2, 1) + alpha * np.interp(pi2_next, grid, J[k+1])

        # Bellman optimality
        J[k, i] = min(J_keep, J_replace)

# Final policy at initial horizon
print("Optimal Value Function at start:")
for pi2, val in zip(grid, J[0]):
    print(f"Belief poor={pi2:.1f}, Value={val:.3f}")
