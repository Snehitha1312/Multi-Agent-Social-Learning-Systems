import numpy as np

p = 0.9   # probability machine stays good
q = 0.5   # probability new machine is poor
c1 = 1.0  # cost when machine is good
c2 = 3.0  # cost when machine is poor
R  = 5.0  # replacement cost
alpha = 0.95  # discount factor
N = 10

# Belief discretization (pi2 = prob machine is poor)
grid = np.linspace(0, 1, 11)  # [0.0, 0.1, ..., 1.0]
n = len(grid)

# Initialize value function
J = np.zeros((N + 1, n))

# Precompute cost arrays
cost_keep = (1 - grid) * c1 + grid * c2
cost_replace = np.full_like(grid, R)

# Precompute next beliefs
pi2_next_keep = grid + (1 - grid) * (1 - p)
pi2_next_replace = np.full_like(grid, q)

# Value iteration backward
for k in range(N - 1, -1, -1):

    J_next_keep = np.interp(pi2_next_keep, grid, J[k + 1])
    J_next_replace = np.interp(pi2_next_replace, grid, J[k + 1])

    J_keep = cost_keep + alpha * J_next_keep
    J_replace = cost_replace + alpha * J_next_replace

    J[k] = np.minimum(J_keep, J_replace)


print("Optimal Value Function at start:")
for pi2, val in zip(grid, J[0]):
    print(f"Belief poor={pi2:.1f}, Value={val:.3f}")
