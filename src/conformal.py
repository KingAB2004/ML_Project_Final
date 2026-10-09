"""Conformal risk control for the critic gate (PLAN Sec. 10.4).

tolerance tau : a turn is a violation when the judge's IP_norm > tau
score s       : cheap in-pipeline scalar from the Critic (its own IP estimate + grounding violation mass)
gate          : release when s <= lambda
risk R(lambda): fraction of calibration turns that would be released at lambda AND are violations
lambda_hat    : the LARGEST lambda whose upper confidence bound on R satisfies UCB(R) <= alpha
                (largest, because among safe thresholds the loosest preserves the most utility)

"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from common import cfg, read_json, write_json


def hoeffding_ucb(r_hat: float, n: int, delta: float) -> float:
    if n <= 0:
        return 1.0
    return min(1.0, r_hat + math.sqrt(math.log(1.0 / delta) / (2.0 * n)))


def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0 if k < n else 1.0
    k = max(0, min(n, k))
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p ** i) * ((1.0 - p) ** (n - i))
    return min(1.0, total)


def bentkus_ucb(r_hat: float, n: int, delta: float) -> float:

    if n <= 0:
        return 1.0
    k = math.ceil(n * r_hat)
    if _pvalue(k, n, 1.0, delta) > delta:
        return 1.0
    lo, hi = r_hat, 1.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if _pvalue(k, n, mid, delta) <= delta:
            hi = mid
        else:
            lo = mid
    return min(1.0, hi)


def _pvalue(k: int, n: int, p: float, delta: float) -> float:
    return math.e * _binom_cdf(k, n, p)


def risk_ucb(r_hat: float, n: int, delta: float, bound: str = "hb") -> float:
    """Hoeffding-Bentkus: the minimum of the two bounds, each valid on its own."""
    if bound == "hoeffding":
        return hoeffding_ucb(r_hat, n, delta)
    if bound == "bentkus":
        return bentkus_ucb(r_hat, n, delta)
    return min(hoeffding_ucb(r_hat, n, delta), bentkus_ucb(r_hat, n, delta))



def nonconformity(ip_pred: float, violation_mass: float, weights: dict | None = None) -> float:
    """s = w1 * critic IP estimate + w2 * grounding violation mass, both already in [0,1]."""
    w = weights or cfg("conformal.score_weights", default={"ip_pred": 0.7, "violation_mass": 0.3})
    s = w.get("ip_pred", 0.7) * float(ip_pred) + w.get("violation_mass", 0.3) * float(violation_mass)
    return max(0.0, min(1.0, s))



@dataclass
class Calibration:
    alpha: float
    tau: float
    delta: float
    n_calibration: int
    lambda_hat: float
    lambda_grid: list[float] = field(default_factory=list)
    risk_curve: list[dict] = field(default_factory=list)
    vacuous: bool = False
    bound: str = "hb"
    score_name: str = "0.7*critic_ip_pred + 0.3*grounding_violation_mass"
    distribution_id: str = "own_generator_v1"
    provenance: dict = field(default_factory=dict)

    def releases(self, score: float) -> bool:
        return score <= self.lambda_hat

    def to_dict(self) -> dict:
        return self.__dict__.copy()

    def save(self, path: str | Path) -> None:
        write_json(path, self.to_dict())

    @staticmethod
    def load(path: str | Path) -> "Calibration":
        d = read_json(path)
        return Calibration(**d)


def risk_curve(scores: Sequence[float], violations: Sequence[bool], lambdas: Sequence[float],
               delta: float, bound: str = "hb") -> list[dict]:
    n = len(scores)
    if n == 0 or n != len(violations):
        raise ValueError("scores and violations must be the same non-zero length")
    curve = []
    for lam in lambdas:
        released = [i for i, s in enumerate(scores) if s <= lam]
        bad = sum(1 for i in released if violations[i])
        r_hat = bad / n
        curve.append({
            "lambda": float(lam),
            "n_released": len(released),
            "release_rate": len(released) / n,
            "risk": r_hat,
            "risk_ucb": risk_ucb(r_hat, n, delta, bound),
        })
    return curve


def default_grid(steps: int | None = None) -> list[float]:
    steps = int(steps or cfg("conformal.lambda_grid_steps", default=41))
    return [i / (steps - 1) for i in range(steps)]


def calibrate(scores: Sequence[float], ip_norms: Sequence[float], alpha: float | None = None,
              tau: float | None = None, delta: float | None = None, bound: str = "hb",
              distribution_id: str = "own_generator_v1", provenance: dict | None = None) -> Calibration:
    alpha = float(cfg("conformal.alpha", default=0.10) if alpha is None else alpha)
    tau = float(cfg("conformal.tau", default=0.50) if tau is None else tau)
    delta = float(cfg("conformal.delta", default=0.10) if delta is None else delta)
    violations = [float(ip) > tau for ip in ip_norms]
    grid = default_grid()
    curve = risk_curve(scores, violations, grid, delta, bound)
    safe = [row for row in curve if row["risk_ucb"] <= alpha]
    vacuous = not safe
    lambda_hat = max(row["lambda"] for row in safe) if safe else min(grid)
    return Calibration(
        alpha=alpha, tau=tau, delta=delta, n_calibration=len(scores), lambda_hat=float(lambda_hat),
        lambda_grid=grid, risk_curve=curve, vacuous=vacuous, bound=bound,
        distribution_id=distribution_id, provenance=provenance or {},
    )


def empirical_violation_rate(ip_norms: Sequence[float], released: Sequence[bool],
                             tau: float | None = None) -> dict:
    tau = float(cfg("conformal.tau", default=0.50) if tau is None else tau)
    rel = [ip for ip, r in zip(ip_norms, released) if r]
    if not rel:
        return {"n_released": 0, "violation_rate": None, "tau": tau}
    return {
        "n_released": len(rel),
        "violation_rate": sum(1 for ip in rel if float(ip) > tau) / len(rel),
        "tau": tau,
    }


def sweep_alphas(scores: Sequence[float], ip_norms: Sequence[float],
                 alphas: Sequence[float] = (0.05, 0.10, 0.20), **kw) -> list[dict]:
    out = []
    for a in alphas:
        cal = calibrate(scores, ip_norms, alpha=a, **kw)
        row = {"alpha": a, "lambda_hat": cal.lambda_hat, "vacuous": cal.vacuous}
        match = next((r for r in cal.risk_curve if r["lambda"] == cal.lambda_hat), None)
        if match:
            row.update({"release_rate": match["release_rate"], "risk": match["risk"],
                        "risk_ucb": match["risk_ucb"]})
        out.append(row)
    return out

#alpha is error bound willing to accept