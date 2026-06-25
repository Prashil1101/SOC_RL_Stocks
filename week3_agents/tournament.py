"""
Week 3 — Round-Robin Tournament Runner
========================================
Competitive Dynamic Pricing | SOC Project

Runs every baseline agent against every other baseline agent (including
mirror matches) for many rounds, and logs results to week3_agents/results/.

Pitfalls addressed (resource sheet):
    - Runs 1,000+ rounds per match so Always-Collude's short-term apparent
      dominance and the longer-run defection incentive both have time to show up.
    - All random seeds are fixed and logged in the output CSV/JSON.
    - Raw results tables are saved (not just printed) for the final report's
      baseline comparison section.

Usage:
    python week3_agents/tournament.py
"""

import sys
import os
import json
import itertools
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from week2_environment.pricing_env import BertrandPricingEnv
from week3_agents.agents import build_standard_roster, BaseAgent

N_ROUNDS = 1000          # resource sheet pitfall #1: must be 1000+
TOURNAMENT_SEED = 7      # logged for reproducibility
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


def run_match(env: BertrandPricingEnv, agent_a: BaseAgent, agent_b: BaseAgent, n_rounds: int = N_ROUNDS):
    """Run one repeated match between two agents for n_rounds, return summary stats."""
    agent_a.reset()
    agent_b.reset()
    env.reset(seed=TOURNAMENT_SEED)

    a_prev_idx, b_prev_idx = None, None
    profits_a, profits_b = [], []
    prices_a, prices_b = [], []

    for _ in range(n_rounds):
        idx_a = agent_a.act(a_prev_idx, b_prev_idx)
        idx_b = agent_b.act(b_prev_idx, a_prev_idx)

        _, (r_a, r_b), _, _, info = env.step((idx_a, idx_b))

        agent_a.update(idx_a, idx_b, r_a)
        agent_b.update(idx_b, idx_a, r_b)

        profits_a.append(r_a)
        profits_b.append(r_b)
        prices_a.append(info["p0"])
        prices_b.append(info["p1"])

        a_prev_idx, b_prev_idx = idx_a, idx_b

    half = n_rounds // 2
    return {
        "agent_a": agent_a.name,
        "agent_b": agent_b.name,
        "n_rounds": n_rounds,
        "avg_profit_a": float(np.mean(profits_a)),
        "avg_profit_b": float(np.mean(profits_b)),
        "avg_profit_a_first_half": float(np.mean(profits_a[:half])),
        "avg_profit_a_second_half": float(np.mean(profits_a[half:])),
        "avg_profit_b_first_half": float(np.mean(profits_b[:half])),
        "avg_profit_b_second_half": float(np.mean(profits_b[half:])),
        "avg_price_a": float(np.mean(prices_a)),
        "avg_price_b": float(np.mean(prices_b)),
    }


def run_tournament():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    env = BertrandPricingEnv(seed=TOURNAMENT_SEED)
    roster = build_standard_roster(env, random_seed=TOURNAMENT_SEED)

    print("=" * 70)
    print("  WEEK 3 — ROUND-ROBIN BASELINE TOURNAMENT")
    print("=" * 70)
    print(f"  Rounds per match : {N_ROUNDS}")
    print(f"  Seed             : {TOURNAMENT_SEED}")
    print(f"  Nash price       : {env.p_nash:.4f}  (action idx {env.nash_action})")
    print(f"  Collusive price  : {env.p_collude:.4f}  (action idx {env.collude_action})")
    print(f"  Agents           : {[a.name for a in roster]}")
    print("=" * 70)

    rows = []
    for agent_a, agent_b in itertools.combinations_with_replacement(roster, 2):
        result = run_match(env, agent_a, agent_b)
        rows.append(result)
        print(
            f"  {result['agent_a']:>14} vs {result['agent_b']:<14} | "
            f"avg_profit_a={result['avg_profit_a']:.4f}  avg_profit_b={result['avg_profit_b']:.4f}"
        )

    df = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "tournament_results.csv")
    df.to_csv(csv_path, index=False)

    meta = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_rounds": N_ROUNDS,
        "seed": TOURNAMENT_SEED,
        "p_nash": env.p_nash,
        "p_collude": env.p_collude,
        "nash_action": env.nash_action,
        "collude_action": env.collude_action,
        "agents": [a.name for a in roster],
    }
    meta_path = os.path.join(RESULTS_DIR, "tournament_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("=" * 70)
    print(f"  Saved raw results -> {csv_path}")
    print(f"  Saved run metadata -> {meta_path}")
    print("=" * 70)
    return df


if __name__ == "__main__":
    run_tournament()
