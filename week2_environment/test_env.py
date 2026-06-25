"""
Week 2 — Unit Tests: BertrandPricingEnv
========================================
Run with:  python week2_environment/test_env.py

Validates:
  1. Nash price output matches analytical formula
  2. Reward is correctly normalised to [0, 1]
  3. Observation space and action space contracts hold
  4. Episode terminates at max_steps
  5. Collude > Nash profit (incentive to collude)
"""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from week2_environment.pricing_env import BertrandPricingEnv
from week1_foundations.nash_derivation import nash_price, profit


PASS = "\033[92m  PASS\033[0m"
FAIL = "\033[91m  FAIL\033[0m"


def run_test(name, condition, detail=""):
    status = PASS if condition else FAIL
    print(f"{status}  {name}" + (f" — {detail}" if detail else ""))
    return condition


def test_nash_price():
    env = BertrandPricingEnv()
    expected = nash_price()
    diff = abs(env.p_nash - expected)
    return run_test(
        "Nash price matches analytical formula",
        diff < 1e-8,
        f"env={env.p_nash:.6f}, expected={expected:.6f}"
    )


def test_collude_gt_nash():
    env = BertrandPricingEnv()
    return run_test(
        "Collusive price > Nash price (incentive to collude)",
        env.p_collude > env.p_nash,
        f"collude={env.p_collude:.4f} > nash={env.p_nash:.4f}"
    )


def test_nash_profit_positive():
    env = BertrandPricingEnv()
    p = env.p_nash
    pi = profit(p, p)
    return run_test(
        "Nash equilibrium profit is positive",
        pi > 0,
        f"π_nash={pi:.6f}"
    )


def test_reset_returns_valid_obs():
    env = BertrandPricingEnv()
    obs, info = env.reset()
    in_space = env.observation_space.contains(obs)
    correct_shape = obs.shape == (2,)
    return run_test(
        "reset() returns obs within observation_space",
        in_space and correct_shape,
        f"obs={obs}, shape={obs.shape}"
    )


def test_step_returns_valid_obs():
    env = BertrandPricingEnv()
    env.reset()
    action = env.action_space.sample()
    obs, rewards, terminated, truncated, info = env.step(action)
    in_space = env.observation_space.contains(obs)
    return run_test(
        "step() returns obs within observation_space",
        in_space,
        f"obs={obs}"
    )


def test_reward_normalised():
    env = BertrandPricingEnv()
    env.reset()
    all_ok = True
    for _ in range(200):
        action = env.action_space.sample()
        _, rewards, _, _, _ = env.step(action)
        r0, r1 = rewards
        if not (0.0 <= r0 <= 1.0 and 0.0 <= r1 <= 1.0):
            all_ok = False
            break
        if env._step >= env.max_steps:
            env.reset()
    return run_test(
        "Rewards normalised to [0, 1] across 200 random steps",
        all_ok
    )


def test_episode_terminates():
    env = BertrandPricingEnv(max_steps=10)
    env.reset()
    truncated = False
    for _ in range(10):
        action = env.action_space.sample()
        _, _, _, truncated, _ = env.step(action)
    return run_test(
        "Episode truncates at max_steps",
        truncated,
        f"truncated={truncated}"
    )


def test_nash_action_closest_to_nash_price():
    env = BertrandPricingEnv()
    nash_idx = env.nash_action
    nash_p_from_grid = env.prices[nash_idx]
    diff = abs(nash_p_from_grid - env.p_nash)
    grid_step = env.prices[1] - env.prices[0]
    return run_test(
        "nash_action maps to price closest to analytical P*",
        diff <= grid_step,
        f"grid={nash_p_from_grid:.4f}, P*={env.p_nash:.4f}, diff={diff:.4f}"
    )


def test_seeded_reproducibility():
    env1 = BertrandPricingEnv(seed=0)
    env2 = BertrandPricingEnv(seed=0)
    obs1, _ = env1.reset()
    obs2, _ = env2.reset()
    return run_test(
        "Same seed → identical reset observations",
        np.allclose(obs1, obs2),
        f"obs1={obs1}, obs2={obs2}"
    )


def test_configurable_params():
    env = BertrandPricingEnv(a=2.0, b=1.5, c=0.3, mc=0.1)
    expected = (2.0 + 1.5 * 0.1) / (2 * 1.5 - 0.3)
    diff = abs(env.p_nash - expected)
    return run_test(
        "Custom demand params produce correct Nash price",
        diff < 1e-8,
        f"env={env.p_nash:.6f}, expected={expected:.6f}"
    )


if __name__ == "__main__":
    print("\n" + "=" * 55)
    print("  WEEK 2 — ENVIRONMENT UNIT TESTS")
    print("=" * 55)

    tests = [
        test_nash_price,
        test_collude_gt_nash,
        test_nash_profit_positive,
        test_reset_returns_valid_obs,
        test_step_returns_valid_obs,
        test_reward_normalised,
        test_episode_terminates,
        test_nash_action_closest_to_nash_price,
        test_seeded_reproducibility,
        test_configurable_params,
    ]

    results = [t() for t in tests]
    n_pass = sum(results)
    n_fail = len(results) - n_pass

    print("=" * 55)
    print(f"  Results: {n_pass}/{len(results)} passed", end="")
    if n_fail == 0:
        print("  ✓  Environment is ready for Week 3.")
    else:
        print(f"\n  {n_fail} test(s) failed — fix environment before proceeding.")
    print("=" * 55 + "\n")
