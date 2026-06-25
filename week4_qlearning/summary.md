# Week 4 Mid-Project Review

## Objective

Implement a tabular Q-learning agent from scratch and verify learning performance.

---

## Learning Algorithm

Q-Learning

Bellman Update:

Q(s,a) ← Q(s,a) + α[r + γ max_a'Q(s',a') − Q(s,a)]

Parameters:

- α = 0.15
- γ = 0.95
- ε start = 1.0
- ε end = 0.05

---

## State Representation

State:

(own_previous_price_index,
 rival_previous_price_index)

Total state space:

50 × 50 = 2500 states

---

## Action Space

50 discrete price levels.

Each action corresponds to a specific market price.

---

## Training Setup

Episodes:

3000

Environment:

BertrandPricingEnv

Training mode:

Self-play

Two independent Q-learning agents interact repeatedly.

---

## Evaluation

Mandatory Gate Check:

Q-Learning vs Random Agent

Evaluation horizon:

1000 rounds

Success criterion:

Average profit(Q-Learning) >
Average profit(Random)

---

## Deliverables

Generated automatically:

- week4_training_log.csv
- qlearning_convergence.png
- qlearning_vs_random.png
- week4_gate_check.json

---

## Mid-Term Status

 Week 1 Complete

 Week 2 Complete

 Week 3 Complete

 Week 4 Complete

Project is ready for Deep Reinforcement Learning extensions in Weeks 5–8.
