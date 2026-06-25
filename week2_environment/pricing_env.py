"""
Week 2 — Custom Gymnasium Environment: Bertrand Pricing Market
==============================================================
Competitive Dynamic Pricing | SOC Project

Implements BertrandPricingEnv — a two-firm differentiated Bertrand duopoly
as an OpenAI Gymnasium environment. This environment is the foundation for
all agent experiments in Weeks 3–8.

Design follows Calvano et al. (QJE 2020) Table 1 parameters.

Key design decisions (per resource sheet pitfall list):
  - Price grid: 50 discrete levels (not 5 — avoids coarse discretisation)
  - Demand parameters: fully configurable (not hard-coded)
  - Observations normalised to [0, 1] from day one
  - Nash price is analytically verifiable (see week1_foundations/)

Usage:
    env = BertrandPricingEnv()
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step((action_0, action_1))
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from week1_foundations.nash_derivation import nash_price, collusive_price, demand, profit


class BertrandPricingEnv(gym.Env):
    """Two-firm differentiated Bertrand duopoly environment.

    Observation space:
        Box([p0_prev_norm, p1_prev_norm], shape=(2,), dtype=float32)
        Both prices normalised to [0, 1] over the price grid.

    Action space:
        MultiDiscrete([n_prices, n_prices])
        Each firm independently selects a discrete price index.

    Reward:
        Tuple (reward_0, reward_1) — profit for each firm, normalised to [0, 1].
        Normalisation prevents Q-value explosion in Week 4.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        a: float = 1.0,
        b: float = 1.0,
        c: float = 0.5,
        mc: float = 0.0,
        n_prices: int = 50,
        p_min: float = 0.0,
        p_max: float = 2.0,
        max_steps: int = 1000,
        seed: int = 42,
    ):
        """
        Args:
            a:        demand intercept
            b:        own-price sensitivity
            c:        cross-price sensitivity (substitutes: 0 < c < b)
            mc:       marginal cost
            n_prices: number of discrete price levels (≥ 20 recommended)
            p_min:    minimum price
            p_max:    maximum price
            max_steps: episode length
            seed:     random seed for reproducibility
        """
        super().__init__()

        assert n_prices >= 20, "Use at least 20 price levels (resource sheet pitfall #1)"
        assert 0 < c < b, "Cross-price effect must satisfy 0 < c < b for valid model"

        # Market parameters
        self.a = a
        self.b = b
        self.c = c
        self.mc = mc
        self.max_steps = max_steps

        # Price grid — discrete actions map to continuous prices
        self.prices = np.linspace(p_min, p_max, n_prices)
        self.n_prices = n_prices
        self.p_min = p_min
        self.p_max = p_max

        # Analytical reference prices
        self.p_nash = nash_price(a, b, c, mc)
        self.p_collude = collusive_price(a, b, c, mc)

        # Max possible profit (for normalisation) — at collusive price, both collude
        self._max_profit = profit(self.p_collude, self.p_collude, mc)
        if self._max_profit <= 0:
            self._max_profit = 1.0  # fallback guard

        # Gymnasium spaces
        # Actions: each firm picks an index into self.prices
        self.action_space = spaces.MultiDiscrete([n_prices, n_prices])

        # Observations: previous round prices (both firms), normalised to [0,1]
        self.observation_space = spaces.Box(
            low=np.float32([0.0, 0.0]),
            high=np.float32([1.0, 1.0]),
            dtype=np.float32,
        )

        # Internal state
        self._step = 0
        self._prev_actions = np.array([0, 0], dtype=np.int32)
        self._rng = np.random.default_rng(seed)

    # ── Core Gymnasium API ────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step = 0
        # Start at random prices
        self._prev_actions = self._rng.integers(0, self.n_prices, size=2)
        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action):
        """Advance one pricing round.

        Args:
            action: array-like of shape (2,) — price indices for firm 0 and firm 1

        Returns:
            obs, rewards, terminated, truncated, info
            rewards is a tuple (r0, r1) — one per firm
        """
        action = np.asarray(action, dtype=np.int32)
        assert self.action_space.contains(action), f"Invalid action: {action}"

        p0 = self.prices[action[0]]
        p1 = self.prices[action[1]]

        # Compute profits
        pi0 = profit(p0, p1, self.mc)
        pi1 = profit(p1, p0, self.mc)

        # Normalise rewards to [0, 1]
        r0 = float(np.clip(pi0 / self._max_profit, 0.0, 1.0))
        r1 = float(np.clip(pi1 / self._max_profit, 0.0, 1.0))

        self._prev_actions = action
        self._step += 1

        terminated = False
        truncated = self._step >= self.max_steps

        obs = self._get_obs()
        info = self._get_info(p0, p1, pi0, pi1)

        return obs, (r0, r1), terminated, truncated, info

    def render(self, mode="human"):
        p0 = self.prices[self._prev_actions[0]]
        p1 = self.prices[self._prev_actions[1]]
        print(
            f"Step {self._step:4d} | "
            f"P0={p0:.3f}  P1={p1:.3f} | "
            f"Nash={self.p_nash:.3f}  Collude={self.p_collude:.3f}"
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_obs(self) -> np.ndarray:
        """Return normalised price observation."""
        return np.array(
            [
                self._prev_actions[0] / (self.n_prices - 1),
                self._prev_actions[1] / (self.n_prices - 1),
            ],
            dtype=np.float32,
        )

    def _get_info(self, p0=None, p1=None, pi0=None, pi1=None) -> dict:
        return {
            "step": self._step,
            "p0": p0,
            "p1": p1,
            "pi0": pi0,
            "pi1": pi1,
            "p_nash": self.p_nash,
            "p_collude": self.p_collude,
        }

    def price_index(self, price: float) -> int:
        """Return the index of the grid price closest to `price`."""
        return int(np.argmin(np.abs(self.prices - price)))

    @property
    def nash_action(self) -> int:
        """Action index corresponding to the Nash equilibrium price."""
        return self.price_index(self.p_nash)

    @property
    def collude_action(self) -> int:
        """Action index corresponding to the collusive price."""
        return self.price_index(self.p_collude)
