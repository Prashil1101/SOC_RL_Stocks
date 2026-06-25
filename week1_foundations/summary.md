# Week 1 — Game Theory Foundations

## Overview

The objective of Week 1 was to analytically derive the Bertrand Nash Equilibrium for a differentiated duopoly and establish a benchmark for later reinforcement learning experiments.

---

## Bertrand Duopoly Model

Two firms compete on price.

Demand for firm i:

Q_i = a - bP_i + cP_j

where:

* a = demand intercept
* b = own-price sensitivity
* c = cross-price sensitivity
* P_i = firm's own price
* P_j = rival firm's price

---

## Profit Function

Profit for firm i:

π_i = (P_i - mc)Q_i

where mc is marginal cost.

---

## Best Response Derivation

Substituting demand into profit:

π_i = (P_i - mc)(a - bP_i + cP_j)

First-order condition:

∂π_i/∂P_i = 0

Result:

P_i*(P_j) = (a + cP_j + bmc)/(2b)

---

## Symmetric Nash Equilibrium

At equilibrium:

P_i = P_j = P*

Therefore:

P* = (a + bmc)/(2b - c)

Using project parameters:

a = 1

b = 1

c = 0.5

mc = 0

Result:

P* = 0.6667

---

## Collusive Benchmark

Joint profit maximization yields:

P_c = (a + bmc)/(2(b-c))

Result:

P_c = 1.0000

---

## Comparison

| Outcome   | Price  |
| --------- | ------ |
| Nash      | 0.6667 |
| Collusive | 1.0000 |

Collusion produces higher profits, creating incentives for cooperative behaviour in repeated interactions.

---

## Key Insight

The gap between Nash and Collusive outcomes motivates the reinforcement learning experiments in later weeks.

The central question becomes:

Can autonomous learning agents discover and sustain prices above the Nash equilibrium?
