import numpy as np

# --- Define MDP Parameters ---
theta = 0.1   # probability of failure
c = 4         # cost of leaving failed machine as is
R = 3         # cost of replacing machine
gamma = 1.0   # no discounting in finite-horizon example

# States: 0 = working, 1 = failed
# Actions: 0 = leave, 1 = replace
P = np.array([
    # Action 0 (leave)
    [[1 - theta, theta],   # from state 0
     [0.0, 1.0]],          # from state 1
    # Action 1 (replace)
    [[1.0, 0.0],           # from state 0
     [1.0, 0.0]]           # from state 1
])

# Costs instead of rewards
C = np.array([
    [0, c],  # costs for action 0 in states 0,1
    [R, R]   # costs for action 1 in states 0,1
])

n_states = 2
n_actions = 2
N = 4  # finite horizon steps

# --- Backward Dynamic Programming (Finite Horizon) ---
J = np.zeros((N+1, n_states))   # cost-to-go
policy = np.zeros((N, n_states), dtype=int)

for k in range(N-1, -1, -1):
    # shape: (n_actions, n_states)
    expected_future = P @ J[k+1]      # matrix multiplication handles dot products
    # broadcast costs (n_actions, n_states)
    total_costs = C + expected_future.T
    # take min over actions (axis=0)
    J[k] = np.min(total_costs, axis=0)
    policy[k] = np.argmin(total_costs, axis=0)

print("Cost-to-go matrix J:")
print(J)
print("\nOptimal Policy (per stage):")
print(policy)
