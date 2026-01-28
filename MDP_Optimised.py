
#vectorization - solving linear systems directly: V=(I−γPπ​)-1 Rπ​
# for fixed policy , Bellman en=qns are just set of linear eq
#for policy evaluation we sole that directly

#policy improvement: uses np.einsum(einstein summation) to compute all Q(s,a) in one go


import numpy as np

# States: 0, 1, 2 | Actions: 0, 1

# Transition probability tensor: P[a, s, s']
P = np.array([
    # Action 0
    [[0.5, 0.5, 0.0],  # from state 0
     [0.0, 0.5, 0.5],  # from state 1
     [0.0, 0.0, 1.0]], # from state 2
    # Action 1
    [[0.0, 1.0, 0.0],  # from state 0
     [0.0, 0.0, 1.0],  # from state 1
     [0.0, 0.0, 1.0]]  # from state 2
])

# Rewards: R[a, s]
R = np.array([
    [5, 0, 0],   # rewards for action 0 in states 0, 1, 2
    [10, 2, 0]   # rewards for action 1 in states 0, 1, 2
])

gamma = 0.9
n_actions, n_states, _ = P.shape

#  Policy Iteration (Fully Vectorized)
policy = np.zeros(n_states, dtype=int)

while True:
    # Policy Evaluation (Vectorized)
    P_pi = P[policy, np.arange(n_states)]  # transition matrix for current policy
    R_pi = R[policy, np.arange(n_states)]  # rewards for current policy
    V = np.linalg.solve(np.eye(n_states) - gamma * P_pi, R_pi)

    # Policy Improvement (Vectorized)
    # Computing Q-values for all actions simultaneously
    Q = R + gamma * np.einsum('ask,k->as', P, V)
    new_policy = np.argmax(Q, axis=0)

    if np.array_equal(new_policy, policy):
        break
    policy = new_policy

print("Optimal Policy (no loops):", policy)
print("Optimal Value Function (no loops):", V)
