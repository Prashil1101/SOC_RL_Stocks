# Competitive Dynamic Pricing using Multi-Agent Reinforcement Learning

## SOC Mid-Term Submission (Weeks 1–4)

---

## Project Overview

This project investigates competitive dynamic pricing in a differentiated Bertrand duopoly using Multi-Agent Reinforcement Learning (MARL).

The objective is to understand how autonomous pricing agents interact in repeated markets, whether they converge to the Bertrand-Nash equilibrium, and whether repeated interaction creates incentives for supra-competitive (collusive) pricing behaviour.

The implementation follows the framework introduced by Calvano et al. (2020), who showed that reinforcement learning algorithms can learn pricing strategies that resemble collusion without explicit communication.

---

## Project Timeline

### Week 1 – Game Theory Foundations

* Analytical derivation of Bertrand Nash Equilibrium
* Best-response function derivation
* Collusive benchmark derivation
* Numerical verification of equilibrium

### Week 2 – Custom Gymnasium Environment

* Built BertrandPricingEnv
* Configurable market parameters
* Reward normalization
* Observation normalization
* Unit testing framework

### Week 3 – Rule-Based Baselines

Implemented:

* Random Agent
* Always Nash Agent
* Always Collude Agent
* Tit For Tat Agent

Conducted round-robin tournament evaluation.

### Week 4 – Tabular Q-Learning

Implemented:

* Hand-coded Bellman update
* Epsilon-greedy exploration
* Self-play training
* Evaluation against Random baseline

---

## Market Model

Demand:

Q_i = a − bP_i + cP_j

Profit:

π_i = (P_i − mc)Q_i

Parameters used:

a = 1.0

b = 1.0

c = 0.5

mc = 0.0

---

## Analytical Results

### Nash Equilibrium

P* = (a + b·mc)/(2b − c)

Result:

P* = 0.6667

### Collusive Price

P_c = (a + b·mc)/(2(b − c))

Result:

P_c = 1.0000

The difference between these outcomes creates an incentive to collude.

---

## Repository Structure

week1_foundations/
week2_environment/
week3_agents/
week4_qlearning/
results/
docs/

---

## Installation

pip install -r requirements.txt

---

## Running

Week 1

python week1_foundations/nash_derivation.py

Week 2

python week2_environment/test_env.py

Week 3

python week3_agents/tournament.py

Week 4

python week4_qlearning/train.py

---

## References

1. Calvano, Calzolari, Denicolò, Pastorello (2020)
2. Sutton and Barto – Reinforcement Learning
3. Watkins and Dayan – Q-Learning
4. Axelrod – The Evolution of Cooperation
