"""
Week 1 — Analytical Nash Equilibrium Derivation
================================================
Competitive Dynamic Pricing | SOC Project

Derives the Bertrand Nash equilibrium price analytically from first principles.
Students must understand this derivation before writing any agent code.

Model (Bertrand duopoly with differentiated products):
  - Two firms i ∈ {0, 1}
  - Linear demand: Q_i = a - b*P_i + c*P_j   (cross-price elasticity c > 0)
  - Marginal cost: mc (symmetric)
  - Profit: π_i = (P_i - mc) * Q_i

Nash Equilibrium:
  Each firm maximises π_i w.r.t. P_i, taking P_j as given.
  ∂π_i/∂P_i = 0  →  Best-response function
  Solve the symmetric fixed point: P_i* = P_j* = P_Nash

Reference: Calvano et al. (QJE 2020), Section 2
"""

import numpy as np


# ── Market parameters (match Calvano et al. Table 1) ──────────────────────────
A = 1.0       # demand intercept
B = 1.0       # own-price sensitivity
C = 0.5       # cross-price sensitivity (substitutes: 0 < C < B)
MC = 0.0      # marginal cost (normalised to 0)
# ──────────────────────────────────────────────────────────────────────────────


def demand(p_own: float, p_rival: float, a=A, b=B, c=C) -> float:
    """Linear demand function for firm i.

    Q_i = a - b*P_i + c*P_j

    Args:
        p_own:   own price
        p_rival: rival's price
    Returns:
        quantity demanded (clamped to 0)
    """
    return max(0.0, a - b * p_own + c * p_rival)


def profit(p_own: float, p_rival: float, mc=MC) -> float:
    """Profit for firm i given own and rival price.

    π_i = (P_i - mc) * Q_i
    """
    return (p_own - mc) * demand(p_own, p_rival)


def best_response(p_rival: float, a=A, b=B, c=C, mc=MC) -> float:
    """Analytical best-response price for firm i given rival price P_j.

    Derivation:
        π_i = (P_i - mc)(a - b*P_i + c*P_j)
        ∂π_i/∂P_i = (a - b*P_i + c*P_j) + (P_i - mc)(-b) = 0
        a - b*P_i + c*P_j - b*P_i + b*mc = 0
        a + c*P_j + b*mc = 2b*P_i
        P_i* = (a + c*P_j + b*mc) / (2b)
    """
    return (a + c * p_rival + b * mc) / (2 * b)


def nash_price(a=A, b=B, c=C, mc=MC) -> float:
    """Compute the symmetric Bertrand Nash equilibrium price analytically.

    At symmetric NE: P* = BR(P*)
        P* = (a + c*P* + b*mc) / (2b)
        2b*P* = a + c*P* + b*mc
        P*(2b - c) = a + b*mc
        P* = (a + b*mc) / (2b - c)

    Existence condition: 2b > c  (own-price effect dominates cross-price)
    """
    assert 2 * b > c, "Existence condition violated: need 2b > c"
    return (a + b * mc) / (2 * b - c)


def collusive_price(a=A, b=B, c=C, mc=MC) -> float:
    """Compute the joint-profit-maximising (collusive) price.

    Both firms maximise π_0 + π_1 jointly.
    By symmetry, solve ∂(π_0+π_1)/∂P_0 = 0 at P_0=P_1=P_c:
        P_c = (a + b*mc) / (2*(b - c))
    """
    assert b > c, "Collusive price undefined when b ≤ c"
    return (a + b * mc) / (2 * (b - c))


def verify_by_iteration(a=A, b=B, c=C, mc=MC, tol=1e-10, max_iter=1000) -> float:
    """Verify Nash price numerically by iterating best-response functions.

    This is NOT how agents learn — it is a sanity check for the analytical formula.
    """
    p0, p1 = 0.5, 0.5
    for _ in range(max_iter):
        p0_new = best_response(p1, a, b, c, mc)
        p1_new = best_response(p0, a, b, c, mc)
        if abs(p0_new - p0) < tol and abs(p1_new - p1) < tol:
            break
        p0, p1 = p0_new, p1_new
    return (p0 + p1) / 2


def print_equilibrium_summary():
    """Print a full equilibrium summary — the Week 1 deliverable."""
    p_nash = nash_price()
    p_collude = collusive_price()
    p_iter = verify_by_iteration()

    q_nash = demand(p_nash, p_nash)
    pi_nash = profit(p_nash, p_nash)
    pi_collude = profit(p_collude, p_collude)

    print("=" * 60)
    print("  WEEK 1 — BERTRAND NASH EQUILIBRIUM DERIVATION")
    print("=" * 60)
    print(f"\nMarket parameters:")
    print(f"  a (demand intercept)     = {A}")
    print(f"  b (own-price effect)     = {B}")
    print(f"  c (cross-price effect)   = {C}")
    print(f"  mc (marginal cost)       = {MC}")

    print(f"\nAnalytical Nash price     P* = (a + b·mc) / (2b - c)")
    print(f"                             = ({A} + {B}·{MC}) / (2·{B} - {C})")
    print(f"                             = {p_nash:.6f}")

    print(f"\nNumerical verification    P* = {p_iter:.6f}  ✓" if abs(p_nash - p_iter) < 1e-8
          else f"\n  WARNING: mismatch — analytical={p_nash:.6f}, numerical={p_iter:.6f}")

    print(f"\nCollusive (joint-max) price P_c = {p_collude:.6f}")
    print(f"  (P_Nash < P_Collude as expected: {p_nash:.4f} < {p_collude:.4f})")

    print(f"\nAt Nash equilibrium:")
    print(f"  Quantity per firm Q*     = {q_nash:.6f}")
    print(f"  Profit per firm   π*     = {pi_nash:.6f}")
    print(f"  Profit at collusion      = {pi_collude:.6f}")
    print(f"  Collusion gain           = {pi_collude - pi_nash:.6f}  (incentive to collude)")

    print(f"\nBest-response function: P_i*(P_j) = (a + c·P_j + b·mc) / 2b")
    print(f"  At P_j = 0.5:  BR = {best_response(0.5):.4f}")
    print(f"  At P_j = P*:   BR = {best_response(p_nash):.4f}  (= P* ✓)")

    print("\n" + "=" * 60)
    print("  KEY INSIGHT")
    print("=" * 60)
    print("  Nash < Collusive price. Firms would earn more by colluding,")
    print("  but Nash is the stable equilibrium — no firm can unilaterally")
    print("  deviate and improve. This tension is what RL agents exploit.")
    print("=" * 60)


if __name__ == "__main__":
    print_equilibrium_summary()
