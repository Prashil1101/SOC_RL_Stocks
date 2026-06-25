# Week 3 Summary

## Objective

Implement rule-based baseline agents and evaluate them in a repeated Bertrand pricing game.

---

## Agents Implemented

### Random Agent

Chooses prices uniformly at random.

Purpose:

Provides a minimum-performance benchmark.

---

### Always Nash Agent

Always chooses the analytical Nash equilibrium price.

Purpose:

Represents rational one-shot competition.

---

### Always Collude Agent

Always chooses the collusive profit-maximizing price.

Purpose:

Represents perfect cooperation.

---

### Tit For Tat Agent

Starts with a cooperative price and then copies the rival's previous action.

Purpose:

Tests whether reciprocal strategies can sustain cooperation.

---

## Tournament Design

All agents play against every other agent.

Rounds per match:

1000

This long horizon allows cooperative and retaliatory dynamics to emerge.

---

## Key Findings

- Random performs worst.
- Nash provides stable benchmark profits.
- Collusion generates the highest joint profits.
- Tit-for-Tat often sustains cooperation against cooperative opponents.

---

## Importance for Week 4

These baselines provide reference points that the Q-learning agent must outperform.
