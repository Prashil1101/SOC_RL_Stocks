"""
Week 4 — Train the Q-Learning Agent + Mid-Project Review Gate Check
========================================================================
Competitive Dynamic Pricing | SOC Project

Trains the tabular QLearningAgent (self-play: two independent Q-learners,
one per firm) against the BertrandPricingEnv, then runs the MANDATORY
mid-project gate check from the resource sheet:

    "The Q-learning agent must consistently beat the Random baseline.
     If it doesn't, the environment or reward function likely has a bug."

Outputs (written to results/):
    - qlearning_convergence.png   : rolling-average profit per firm over training
    - qlearning_vs_random.png     : Q-learning agent's profit vs a Random opponent
    - week4_training_log.csv      : per-episode summary stats
    - week4_gate_check.json       : pass/fail result of the mandatory mid-project gate

Usage:
    python week4_qlearning/train.py
"""

import sys
import os
import json
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from week2_environment.pricing_env import BertrandPricingEnv
from week3_agents.agents import RandomAgent
from week4_qlearning.q_agent import QLearningAgent

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
SEED = 42
N_PRICES = 50
N_EPISODES = 3000          # one "episode" here == one pricing round (env max_steps controls truncation)
N_EVAL_ROUNDS = 1000        # rounds used for the post-training gate check (resource sheet: 1000+)


def train_self_play(n_episodes: int = N_EPISODES, seed: int = SEED):
    """Train two independent Q-learning agents against each other (self-play)."""
    env = BertrandPricingEnv(n_prices=N_PRICES, max_steps=n_episodes, seed=seed)

    agent_0 = QLearningAgent(n_prices=N_PRICES, total_episodes=n_episodes, seed=seed)
    agent_1 = QLearningAgent(n_prices=N_PRICES, total_episodes=n_episodes, seed=seed + 1)

    obs, info = env.reset(seed=seed)
    own_idx0, own_idx1 = int(env._prev_actions[0]), int(env._prev_actions[1])

    log_rows = []
    for ep in range(n_episodes):
        state = (own_idx0, own_idx1)               # agent_0's view: (own, rival)
        state_rev = (own_idx1, own_idx0)            # agent_1's view: (own, rival)

        a0 = agent_0.act(*state)
        a1 = agent_1.act(*state_rev)

        _, (r0, r1), terminated, truncated, info = env.step((a0, a1))

        next_state = (a0, a1)
        next_state_rev = (a1, a0)

        agent_0.update(state, a0, r0, next_state)
        agent_1.update(state_rev, a1, r1, next_state_rev)

        agent_0.decay_epsilon()
        agent_1.decay_epsilon()

        own_idx0, own_idx1 = a0, a1

        log_rows.append({
            "episode": ep,
            "p0": info["p0"],
            "p1": info["p1"],
            "r0": r0,
            "r1": r1,
            "epsilon": agent_0.epsilon,
        })

        if truncated:
            obs, info = env.reset(seed=seed + ep)
            own_idx0, own_idx1 = int(env._prev_actions[0]), int(env._prev_actions[1])

    return env, agent_0, agent_1, pd.DataFrame(log_rows)


def evaluate_vs_random(env: BertrandPricingEnv, q_agent: QLearningAgent, n_rounds: int = N_EVAL_ROUNDS, seed: int = SEED):
    """Mandatory mid-project gate check: Q-learning agent (greedy) vs Random baseline."""
    random_agent = RandomAgent(n_prices=env.n_prices, seed=seed)
    random_agent.reset()

    env.reset(seed=seed)
    own_idx, rival_idx = 0, 0
    q_profits, random_profits = [], []

    saved_epsilon = q_agent.epsilon
    q_agent.epsilon = 0.0  # pure exploitation during evaluation

    for _ in range(n_rounds):
        a_q = q_agent.act(own_idx, rival_idx)
        a_r = random_agent.act(rival_idx, own_idx)

        _, (r_q, r_r), _, _, info = env.step((a_q, a_r))
        q_profits.append(r_q)
        random_profits.append(r_r)
        own_idx, rival_idx = a_q, a_r

    q_agent.epsilon = saved_epsilon

    return {
        "avg_profit_qlearning": float(np.mean(q_profits)),
        "avg_profit_random": float(np.mean(random_profits)),
        "qlearning_beats_random": bool(np.mean(q_profits) > np.mean(random_profits)),
        "n_rounds": n_rounds,
    }


def make_plots(log_df: pd.DataFrame, env: BertrandPricingEnv, gate_result: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    window = max(10, len(log_df) // 100)

    # Convergence plot: rolling avg profit per firm vs Nash/collusive reference profit
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(log_df["episode"], log_df["r0"].rolling(window).mean(), label="Firm 0 (Q-learning)")
    ax.plot(log_df["episode"], log_df["r1"].rolling(window).mean(), label="Firm 1 (Q-learning)")
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Rolling avg. normalised profit (window={window})")
    ax.set_title("Week 4 — Q-Learning Self-Play Convergence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "qlearning_convergence.png"), dpi=150)
    plt.close(fig)

    # Gate check bar plot
    fig, ax = plt.subplots(figsize=(5, 5))
    labels = ["Q-Learning\n(greedy)", "Random"]
    values = [gate_result["avg_profit_qlearning"], gate_result["avg_profit_random"]]
    colors = ["#2a6f97", "#a3a3a3"]
    ax.bar(labels, values, color=colors)
    ax.set_ylabel("Avg. normalised profit per round")
    ax.set_title(f"Mid-Project Gate Check ({gate_result['n_rounds']} rounds)")
    for i, v in enumerate(values):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center")
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS_DIR, "qlearning_vs_random.png"), dpi=150)
    plt.close(fig)


def run():
    print("=" * 70)
    print("  WEEK 4 — TABULAR Q-LEARNING TRAINING")
    print("=" * 70)
    print(f"  n_prices    : {N_PRICES}")
    print(f"  episodes    : {N_EPISODES}")
    print(f"  eval rounds : {N_EVAL_ROUNDS}")
    print("=" * 70)

    env, agent_0, agent_1, log_df = train_self_play()

    print(f"\nTraining complete. Final epsilon: {agent_0.epsilon:.4f}")
    print(f"Nash price reference     : {env.p_nash:.4f}")
    print(f"Collusive price reference: {env.p_collude:.4f}")

    last_window = log_df.tail(max(50, len(log_df) // 20))
    print(f"\nFinal-window avg price (firm 0): {env.prices[int(round(last_window['p0'].mean() / env.prices[1] if env.prices[1] else 0))] if False else last_window['p0'].mean():.4f}")
    print(f"Final-window avg price (firm 1): {last_window['p1'].mean():.4f}")

    print("\nRunning MANDATORY mid-project gate check (Q-learning vs Random)...")
    gate_result = evaluate_vs_random(env, agent_0)
    print(f"  Q-learning avg profit : {gate_result['avg_profit_qlearning']:.4f}")
    print(f"  Random avg profit     : {gate_result['avg_profit_random']:.4f}")
    if gate_result["qlearning_beats_random"]:
        print("  GATE CHECK: PASS — Q-learning consistently beats Random. ✓")
    else:
        print("  GATE CHECK: FAIL — investigate env/reward function before Week 5. ✗")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    log_df.to_csv(os.path.join(RESULTS_DIR, "week4_training_log.csv"), index=False)

    gate_record = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "n_episodes": N_EPISODES,
        **gate_result,
        "p_nash": env.p_nash,
        "p_collude": env.p_collude,
    }
    with open(os.path.join(RESULTS_DIR, "week4_gate_check.json"), "w") as f:
        json.dump(gate_record, f, indent=2)

    make_plots(log_df, env, gate_result)

    print(f"\nSaved training log -> {os.path.join(RESULTS_DIR, 'week4_training_log.csv')}")
    print(f"Saved gate check    -> {os.path.join(RESULTS_DIR, 'week4_gate_check.json')}")
    print(f"Saved plots         -> {RESULTS_DIR}/qlearning_convergence.png, qlearning_vs_random.png")
    print("=" * 70)

    return gate_record


if __name__ == "__main__":
    run()
