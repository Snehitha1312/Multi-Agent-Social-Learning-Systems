import numpy as np

# Hidden state space: 0 = RedMajority, 1 = BlueMajority
states = ["RedMajority", "BlueMajority"]

# Transition matrix (P). Static hidden state -> Identity
P = np.eye(2)

# Observation probabilities P(y|x)
# Row = hidden state, Col = observation (red=0, blue=1)
B = np.array([
    [0.7, 0.3],  # If Red majority
    [0.3, 0.7]   # If Blue majority
])

# Initial public belief π0
pi = np.array([0.6, 0.4])  # [Red, Blue]

# Cost structure: 0 if correct, 1 if wrong
C = np.array([[0, 1],   # Action = Red
              [1, 0]])  # Action = Blue

def private_belief_update(pi, obs):
    """
    Implements formula (2.1): ηk = Byk P^T pi / (1^T Byk P^T pi)
    """
    Byk = np.diag(B[:, obs])   # diag(P(y|x))
    num = Byk @ P.T @ pi
    den = np.sum(num)
    return num / den

def myopic_action(eta):
    """
    Implements formula (2.2): ak = argmin c_a^T ηk
    """
    costs = C @ eta
    return np.argmin(costs)  # 0=Red, 1=Blue

def public_belief_update(pi, action):
    """
    Implements formula (2.3): πk = T(π, a)
    """
    # Compute likelihood of action under each state
    R = np.zeros((2, 2))
    for i in range(2):  # hidden state
        # Compute agent's private observation distribution under state i
        probs = []
        for obs in [0, 1]:  # red=0, blue=1
            eta = private_belief_update(pi, obs)
            act = myopic_action(eta)
            if act == action:
                probs.append(B[i, obs])
            else:
                probs.append(0.0)
        R[i, i] = sum(probs)

    num = R @ P.T @ pi
    den = np.sum(num)
    return num / den

# Simulate a sequence of agents
np.random.seed(42)
true_state = 0  # Assume RedMajority is true
num_agents = 5

for k in range(1, num_agents + 1):
    # Agent draws ball
    obs = np.random.choice([0, 1], p=B[true_state])  # 0=red,1=blue

    # Private belief update
    eta = private_belief_update(pi, obs)

    # Myopic action
    action = myopic_action(eta)

    # Public belief update
    pi = public_belief_update(pi, action)

    print(f"Agent {k}: drew {['Red','Blue'][obs]}, "
          f"private belief={eta.round(3)}, "
          f"action={['Red','Blue'][action]}, "
          f"new public belief={pi.round(3)}")
