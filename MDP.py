# Vπ(s)=R(s,π(s))+ γ ∑​P(s′∣s,π(s))Vπ(s′) - evaluation
# Q(s,a)=R(s,a)+γs′∑​P(s′∣s,a)V(s′) - improvement

import numpy as np

# States: 0,1,2  | Actions: 0,1
P = np.array([
    # action 0
    [[0.5, 0.5, 0.0],
     [0.0, 0.5, 0.5],
     [0.0, 0.0, 1.0]],
    # action 1
    [[0.0, 1.0, 0.0],
     [0.0, 0.0, 1.0],
     [0.0, 0.0, 1.0]]
])

R = np.array([
    [5, 0, 0],   # rewards for action 0 in states 0,1,2 (expected immediate reward)
    [10, 2, 0]   # rewards for action 1 in states 0,1,2
])
gamma = 0.9 #discount factor- gamma=0.9 means future rewards are worth 90% as much as present ones.
n_states = 3
n_actions = 2

# Policy Iteration
policy = np.zeros(n_states, dtype=int)  # initial policy: choose action 0 in all states

while True:
    # Policy Evaluation - If I start in state s and follow my fixed policy forever, how much reward will I get in total
    V = np.zeros(n_states)
    for _ in range(100):  # iterative evaluation until convergence
        new_V = np.zeros_like(V)
        for s in range(n_states):
            a = policy[s]
            new_V[s] = R[a, s] + gamma * np.dot(P[a, s], V)
        if np.max(np.abs(new_V - V)) < 1e-6:
            break
        V = new_V

    # Policy Improvement -Using the computed true value function for curr policy, see if we can improve the policy.
    stable = True
    for s in range(n_states):
        old_action = policy[s]
        action_values = np.zeros(n_actions)
        for a in range(n_actions):
            action_values[a] = R[a, s] + gamma * np.dot(P[a, s], V)
        policy[s] = np.argmax(action_values)
        if policy[s] != old_action:
            stable = False

    if stable:   # if no policy changed for any state, we reached the optimal policy
        break

print("Optimal Policy (with loops):", policy)
print("Optimal Value Function (with loops):", V)
