from banditenv import BanditEnv, frequency
import numpy as np
from typing import Optional, Dict, Any

def mabsolver(
    env: BanditEnv,
    steps: int,
    epsilon: float = 0.1,
    init_value: float = 0.0,
    seed: Optional[int] = None,
    method_name: str = 'epsilon_greedy',
) -> Dict[str, Any]:
    """
    epsilon-greedy with incremental mean updates:
        Q_{t+1}(a) = Q_t(a) + (R_t - Q_t(a)) / N_t(a)
    """
    assert steps > 0
    assert 0.0 <= epsilon <= 1.0

    rng = np.random.default_rng(seed)

    K = env.n_arms
    Q = np.full(K, float(init_value), dtype=float) # Return a new array of given shape and type, filled with fill_value.
    N = np.zeros(K, dtype=int)
    evals = np.zeros(K, dtype=float)

    actions = np.empty(steps, dtype=int)
    rewards = np.empty(steps, dtype=float)
    q_values = np.empty((steps, K), dtype=float)

    for t in range(1, steps):

        if method_name == 'ucb':
            # TODO: implement UCB action selection
        elif method_name == 'epsilon_greedy':
            # TODO: implement epsilon-greedy action selection

        r, info = env.step(a)

        # TODO: Update Statistics like N and Q

        # Save the logs
        actions[t] = a
        rewards[t] = r
        q_values[t] = Q

    return {
        "actions": actions,
        "rewards": rewards,
        "q_values": q_values,
        "counts": N,
        "final_Q": Q,
    }


# Example usage
if __name__ == "__main__":
    env = BanditEnv(
        n_arms=10,
        dist="gaussian",
        nonstationary=False,
        seed=123
    )

    logs = mabsolver(env, steps=5000, epsilon=0.1, init_value=0.0, seed=42, method_name='ucb')

    actions = logs["actions"]

    print('frequency of actions: ', frequency(actions))