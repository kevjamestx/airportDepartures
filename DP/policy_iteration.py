import numpy as np

class PolicyIteration:
    def __init__(self, states, actions, transition_prob, rewards, gamma=1, theta=1e-6):
        """
        Initialize Policy Iteration solver.
        :param states: List of all states.
        :param actions: List of all actions.
        :param transition_prob: Dict P(s' | s, a) -> Probability.
        :param rewards: Dict R(s, a) -> Immediate reward.
        :param gamma: Discount factor.
        :param theta: Convergence threshold.
        """
        self.states = states
        self.actions = actions
        self.P = transition_prob
        self.R = rewards
        self.gamma = gamma
        self.theta = theta
        self.policy = {s: np.random.choice(actions) for s in states}  # Random initial policy
        self.V = {s: 0 for s in states}  # Initialize state values to zero

    def policy_evaluation(self):
        """Evaluate the current policy by solving for V(s)."""
        while True:
            delta = 0
            for s in self.states:
                v = self.V[s]
                a = self.policy[s]
                self.V[s] = sum(self.P[(s, a, s_next)] * (self.R[(s, a)] + self.gamma * self.V[s_next])
                                for s_next in self.states)
                delta = max(delta, abs(v - self.V[s]))
            if delta < self.theta:
                break

    def policy_improvement(self):
        """Improve the policy by making it greedy w.r.t V(s)."""
        policy_stable = True
        for s in self.states:
            old_action = self.policy[s]
            self.policy[s] = max(self.actions, key=lambda a: sum(
                self.P[(s, a, s_next)] * (self.R[(s, a)] + self.gamma * self.V[s_next])
                for s_next in self.states
            ))
            if old_action != self.policy[s]:
                policy_stable = False
        return policy_stable

    def solve(self):
        """Run policy iteration until convergence."""
        while True:
            self.policy_evaluation()
            if self.policy_improvement():
                break
        return self.policy, self.V

# Example usage (you define these based on your Markov chain)
states = [...]  # Define your state space
actions = [...]  # Define your action space
transition_prob = {(...): ...}  # Define transition probabilities P(s' | s, a)
rewards = {(...): ...}  # Define rewards R(s, a)

solver = PolicyIteration(states, actions, transition_prob, rewards)
optimal_policy, value_function = solver.solve()
print("Optimal Policy:", optimal_policy)
print("Value Function:", value_function)
