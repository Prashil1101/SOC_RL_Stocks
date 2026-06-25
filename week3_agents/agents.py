"""
Week 3 — Rule-Based Baseline Agents
====================================
Competitive Dynamic Pricing | SOC Project

Implements four classical rule-based strategies for the BertrandPricingEnv,
to be run BEFORE any RL agent is built (per resource sheet: "baselines
validate the environment AND give the RL agent something meaningful to beat").

Agents implemented:
    - RandomAgent       : picks a uniformly random price index every round
    - AlwaysNashAgent    : always prices at the analytical Nash price
    - AlwaysColludeAgent : always prices at the analytical collusive price
    - TitForTatAgent     : mirrors the rival's previous price (Axelrod, 1984)

Pitfalls addressed (resource sheet):
    - Random agent is ALWAYS seeded and the seed is logged (pitfall: irreproducible results)
    - Always-Collude is evaluated over 1000+ rounds, not just a few, since its
      apparent short-term dominance does not reflect the defection incentive
      that emerges over a longer horizon (see tournament.py)
"""

from __future__ import annotations

import numpy as np


class BaseAgent:
    """Common interface every baseline agent implements.

    Subclasses must implement `act`. `update` is a no-op hook for agents
    that don't learn (kept so the interface matches the Week 4 QLearningAgent
    and tournament.py can treat every agent identically).
    """

    name: str = "BaseAgent"

    def reset(self):
        """Called at the start of every new tournament match."""
        pass

    def act(self, own_prev_idx: int | None, rival_prev_idx: int | None) -> int:
        """Return a discrete price index for this round.

        Args:
            own_prev_idx:   this agent's own price index last round (None on round 0)
            rival_prev_idx: rival's price index last round (None on round 0)
        """
        raise NotImplementedError

    def update(self, own_idx: int, rival_idx: int, reward: float):
        """Hook for learning agents. No-op for rule-based agents."""
        pass

    def __repr__(self):
        return self.name


class RandomAgent(BaseAgent):
    """Prices uniformly at random every round. Always seeded for reproducibility."""

    name = "Random"

    def __init__(self, n_prices: int, seed: int):
        self.n_prices = n_prices
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self):
        # Re-seed on every match so tournament results are reproducible run-to-run.
        self._rng = np.random.default_rng(self.seed)

    def act(self, own_prev_idx, rival_prev_idx) -> int:
        return int(self._rng.integers(0, self.n_prices))


class AlwaysNashAgent(BaseAgent):
    """Always prices at the analytical Bertrand-Nash price index."""

    name = "AlwaysNash"

    def __init__(self, nash_action_idx: int):
        self.nash_action_idx = nash_action_idx

    def act(self, own_prev_idx, rival_prev_idx) -> int:
        return self.nash_action_idx


class AlwaysColludeAgent(BaseAgent):
    """Always prices at the analytical joint-profit-maximising (collusive) price index.

    Note (resource sheet pitfall): this strategy only "wins" if BOTH firms play it.
    Against any rival that ever undercuts, the defection incentive means this
    strategy is exploitable — run tournament.py for 1000+ rounds to see this.
    """

    name = "AlwaysCollude"

    def __init__(self, collude_action_idx: int):
        self.collude_action_idx = collude_action_idx

    def act(self, own_prev_idx, rival_prev_idx) -> int:
        return self.collude_action_idx


class TitForTatAgent(BaseAgent):
    """Mirrors the rival's previous price (Axelrod, 1984).

    Round 0 (no history yet): starts cooperatively at the collusive price index,
    matching the "nice" property Axelrod identified as essential to TFT's success.
    """

    name = "TitForTat"

    def __init__(self, collude_action_idx: int):
        self.collude_action_idx = collude_action_idx

    def act(self, own_prev_idx, rival_prev_idx) -> int:
        if rival_prev_idx is None:
            return self.collude_action_idx
        return int(rival_prev_idx)


def build_standard_roster(env, random_seed: int = 7) -> list[BaseAgent]:
    """Convenience constructor: build the four standard Week 3 agents
    against a given BertrandPricingEnv instance (so Nash/collude actions
    are derived from the env, never hard-coded).
    """
    return [
        RandomAgent(n_prices=env.n_prices, seed=random_seed),
        AlwaysNashAgent(nash_action_idx=env.nash_action),
        AlwaysColludeAgent(collude_action_idx=env.collude_action),
        TitForTatAgent(collude_action_idx=env.collude_action),
    ]
