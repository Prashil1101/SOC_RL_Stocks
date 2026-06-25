"""
Week 4 — Tabular Q-Learning Agent
====================================
Competitive Dynamic Pricing | SOC Project

Hand-coded Q-learning agent with an epsilon-greedy policy and the classic
Bellman update — implemented from scratch (NO Stable-Baselines3, per the
resource sheet: "students who don't hand-code the Bellman update at least
once will treat DQN and PPO as black boxes forever").

State representation:
    State = (own_prev_price_idx, rival_prev_price_idx)
    i.e. the discrete price indices both firms played last round.
    This matches the env's observation (normalised versions of the same
    two indices) but keeps the Q-table indexable by integers directly.

Update rule (Watkins & Dayan, 1992 / Sutton & Barto Ch. 6):
    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

Pitfalls addressed (resource sheet):
    - Epsilon starts at 1.0 and decays SLOWLY to 0.05 over 80% of training
      episodes (not abruptly) — premature exploitation is flagged as the
      #1 reason Q-learning fails to converge.
    - Q-table size is bounded by (n_prices ** 3): state=(n_prices, n_prices),
      action=n_prices. For n_prices=50 that's 125,000 entries per agent —
      manageable, but the resource sheet's bucketing pitfall is noted below
      if a much finer grid is ever used.
    - Rewards arriving from BertrandPricingEnv are already normalised to
      [0, 1] by the environment itself (Week 2), so no extra scaling is
      needed here — this is exactly the fix for the "reward not scaled"
      pitfall called out in the Week 4 sheet.
"""

from __future__ import annotations

import numpy as np


class QLearningAgent:
    """Tabular Q-learning agent for the two-firm BertrandPricingEnv.

    Args:
        n_prices:        number of discrete price levels (== action space size per firm)
        alpha:           learning rate
        gamma:           discount factor
        epsilon_start:   initial exploration rate (should be 1.0)
        epsilon_end:     final exploration rate (resource sheet: 0.05, not lower)
        epsilon_decay_frac: fraction of TOTAL planned episodes over which epsilon
                            decays linearly from epsilon_start to epsilon_end
                            (resource sheet: decay over 80% of training)
        total_episodes:  total planned training episodes, used to compute the
                          per-episode decay step
        seed:            RNG seed for reproducible exploration
    """

    name = "QLearning"

    def __init__(
        self,
        n_prices: int,
        alpha: float = 0.15,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_frac: float = 0.8,
        total_episodes: int = 1,
        seed: int = 0,
    ):
        self.n_prices = n_prices
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end

        decay_episodes = max(1, int(epsilon_decay_frac * total_episodes))
        self._epsilon_step = (epsilon_start - epsilon_end) / decay_episodes
        self._decay_episodes = decay_episodes

        # Q-table: shape (n_prices, n_prices, n_prices)
        #   axis 0: own previous price index   (state component 1)
        #   axis 1: rival previous price index  (state component 2)
        #   axis 2: action (this round's price index)
        self.q_table = np.zeros((n_prices, n_prices, n_prices), dtype=np.float64)

        self._rng = np.random.default_rng(seed)

        # Diagnostics
        self.episode_count = 0
        self.update_count = 0

    def reset(self):
        """Reset exploration rate and episode counters for a fresh training run.

        NOTE: does NOT clear the Q-table — call `agent.q_table[:] = 0` explicitly
        if you want a truly fresh agent. This split lets you resume training
        without losing learned values, while still resetting epsilon if desired.
        """
        self.epsilon = self.epsilon_start
        self.episode_count = 0

    def decay_epsilon(self):
        """Call once per episode. Linear decay over `_decay_episodes`, then floor."""
        self.episode_count += 1
        if self.episode_count <= self._decay_episodes:
            self.epsilon = max(self.epsilon_end, self.epsilon - self._epsilon_step)
        else:
            self.epsilon = self.epsilon_end

    def act(self, own_prev_idx: int, rival_prev_idx: int) -> int:
        """Epsilon-greedy action selection over the Q-table row for the current state."""
        if self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_prices))
        q_row = self.q_table[own_prev_idx, rival_prev_idx]
        # Break ties randomly rather than always picking the first max index —
        # avoids the agent getting stuck pricing at index 0 early in training.
        best = np.flatnonzero(q_row == q_row.max())
        return int(self._rng.choice(best))

    def update(self, state, action: int, reward: float, next_state):
        """Bellman update:

            Q(s,a) <- Q(s,a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s,a) ]

        Args:
            state:      (own_prev_idx, rival_prev_idx) BEFORE this action
            action:     the price index chosen this round
            reward:     normalised profit received ([0, 1], from the env)
            next_state: (own_idx, rival_idx) AFTER this action — i.e. this
                        round's own/rival indices, which become next round's state
        """
        s0, s1 = state
        ns0, ns1 = next_state

        current_q = self.q_table[s0, s1, action]
        best_next_q = self.q_table[ns0, ns1].max()
        td_target = reward + self.gamma * best_next_q
        td_error = td_target - current_q

        self.q_table[s0, s1, action] = current_q + self.alpha * td_error
        self.update_count += 1

    def greedy_action(self, own_prev_idx: int, rival_prev_idx: int) -> int:
        """Pure exploitation — used for evaluation, not training."""
        q_row = self.q_table[own_prev_idx, rival_prev_idx]
        return int(np.argmax(q_row))

    def __repr__(self):
        return f"QLearningAgent(epsilon={self.epsilon:.3f}, updates={self.update_count})"
