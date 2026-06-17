"""Real statistics for keyword/search-term decisions.

Every function here uses scipy / statsmodels and is unit-meaningful:

* :func:`two_proportion_test` - statsmodels ``proportions_ztest`` of a term's conversion
  rate vs the rest of its ad group, returning p-value, effect size and direction.
* :func:`wilson_interval` - Wilson score CI for a binomial proportion (honest at low n).
* :func:`empirical_bayes_shrinkage` - Beta-Binomial Empirical-Bayes shrinkage of each
  term's conversion rate toward the group prior (the low-volume guardrail).
* :func:`decide` - configurable decision function acting at a chosen confidence level,
  tagging each row PRUNE / PROMOTE / NEGATIVE / HOLD-INCUBATING / LOW-VOLUME.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.proportion import proportions_ztest

Direction = Literal["above", "below", "flat"]


# --------------------------------------------------------------------------- #
# Two-proportion z-test
# --------------------------------------------------------------------------- #
def two_proportion_test(
    term_conv: int, term_clicks: int, rest_conv: int, rest_clicks: int
) -> tuple[float, float, Direction]:
    """Two-proportion z-test of term CVR vs the rest-of-group CVR.

    Returns ``(p_value, effect_size, direction)`` where ``effect_size`` is the
    absolute difference in conversion rates (term minus rest) and ``direction`` is
    whether the term converts above or below the group.
    """
    if term_clicks == 0 or rest_clicks == 0:
        return 1.0, 0.0, "flat"

    count = np.array([term_conv, rest_conv])
    nobs = np.array([term_clicks, rest_clicks])
    # Guard against degenerate all-zero / all-one cases that make the z-test undefined.
    if count.sum() == 0 or (count == nobs).all():
        return 1.0, 0.0, "flat"

    try:
        _, p_value = proportions_ztest(count, nobs)
    except Exception:
        return 1.0, 0.0, "flat"
    if np.isnan(p_value):
        p_value = 1.0

    term_rate = term_conv / term_clicks
    rest_rate = rest_conv / rest_clicks
    effect = term_rate - rest_rate
    direction: Direction = "above" if effect > 0 else ("below" if effect < 0 else "flat")
    return float(p_value), float(effect), direction


# --------------------------------------------------------------------------- #
# Wilson confidence interval
# --------------------------------------------------------------------------- #
def wilson_interval(successes: int, trials: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Much better than the normal-approximation interval when ``trials`` is small or the
    proportion is near 0/1, which is exactly the regime for long-tail search terms.
    """
    if trials == 0:
        return 0.0, 1.0
    z = scipy_stats.norm.ppf(1 - (1 - confidence) / 2)
    phat = successes / trials
    denom = 1 + z**2 / trials
    centre = (phat + z**2 / (2 * trials)) / denom
    margin = (z / denom) * np.sqrt(phat * (1 - phat) / trials + z**2 / (4 * trials**2))
    return float(max(0.0, centre - margin)), float(min(1.0, centre + margin))


# --------------------------------------------------------------------------- #
# Empirical-Bayes / Beta-Binomial shrinkage
# --------------------------------------------------------------------------- #
@dataclass
class BetaPrior:
    """Beta(alpha, beta) prior fitted from the population of terms."""

    alpha: float
    beta: float

    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


def fit_beta_prior(conversions: np.ndarray, clicks: np.ndarray) -> BetaPrior:
    """Method-of-moments fit of a Beta prior to the per-term conversion rates.

    We weight each term by its clicks so high-volume terms inform the prior more.
    Falls back to a weakly-informative prior when there is not enough signal.
    """
    clicks = np.asarray(clicks, dtype=float)
    conversions = np.asarray(conversions, dtype=float)
    mask = clicks > 0
    if mask.sum() < 3:
        return BetaPrior(alpha=1.0, beta=19.0)  # weak prior ~5% CVR

    rates = conversions[mask] / clicks[mask]
    w = clicks[mask]
    mu = np.average(rates, weights=w)
    var = np.average((rates - mu) ** 2, weights=w)

    if var <= 1e-9 or mu <= 0 or mu >= 1:
        # No spread (or degenerate) -> fall back to a weak prior centred on mu.
        mu = min(max(mu, 1e-3), 1 - 1e-3)
        strength = 20.0
        return BetaPrior(alpha=mu * strength, beta=(1 - mu) * strength)

    # Method of moments for a Beta distribution.
    common = mu * (1 - mu) / var - 1
    alpha = max(mu * common, 1e-3)
    beta = max((1 - mu) * common, 1e-3)
    return BetaPrior(alpha=alpha, beta=beta)


def empirical_bayes_shrinkage(
    conversions: np.ndarray, clicks: np.ndarray, prior: BetaPrior | None = None
) -> np.ndarray:
    """Posterior-mean conversion rate per term, shrunk toward the group prior.

    ``shrunk_rate = (alpha + conv) / (alpha + beta + clicks)``. Low-click terms are
    pulled strongly toward ``prior.mean``; high-click terms keep their observed rate.
    This is the guardrail that stops a 1/3 fluke from being promoted or a 0/4 term
    from being prematurely pruned.
    """
    conversions = np.asarray(conversions, dtype=float)
    clicks = np.asarray(clicks, dtype=float)
    if prior is None:
        prior = fit_beta_prior(conversions, clicks)
    return (prior.alpha + conversions) / (prior.alpha + prior.beta + clicks)


# --------------------------------------------------------------------------- #
# Decision function
# --------------------------------------------------------------------------- #
@dataclass
class DecisionConfig:
    """Thresholds for the decision function."""

    confidence: float = 0.95          # e.g. 0.80 to act more aggressively
    min_clicks: int = 15              # below this -> LOW-VOLUME / HOLD-INCUBATING
    incubation_clicks: int = 30       # terms below this with no signal incubate
    negative_min_cost: float = 25.0   # cost gate for hard NEGATIVE recommendation
    promote_min_shrunk_uplift: float = 0.02  # shrunk CVR must beat group mean by this


def decide(df: pd.DataFrame, cfg: DecisionConfig | None = None) -> pd.DataFrame:
    """Tag each search term with a recommendation.

    Adds columns: ``cvr``, ``p_value``, ``effect``, ``direction``, ``wilson_low``,
    ``wilson_high``, ``shrunk_cvr``, ``group_mean_cvr``, ``recommendation``, ``reason``.

    Tags
    ----
    PROMOTE          : significantly above group AND shrunk CVR beats the mean -> harvest as exact.
    PRUNE            : significantly below group with enough volume -> pause / lower bid.
    NEGATIVE         : meaningful spend, zero conversions, significant -> add as negative.
    HOLD-INCUBATING  : not enough clicks yet, but spending - keep watching.
    LOW-VOLUME       : barely any clicks - no action, shrink toward prior.
    """
    cfg = cfg or DecisionConfig()
    alpha = 1 - cfg.confidence

    out = df.copy()
    total_conv = out["conversions"].sum()
    total_clicks = out["clicks"].sum()
    group_mean = (total_conv / total_clicks) if total_clicks > 0 else 0.0

    prior = fit_beta_prior(out["conversions"].to_numpy(), out["clicks"].to_numpy())
    out["shrunk_cvr"] = empirical_bayes_shrinkage(
        out["conversions"].to_numpy(), out["clicks"].to_numpy(), prior
    )
    out["group_mean_cvr"] = group_mean

    p_values, effects, directions = [], [], []
    wlo, whi, cvrs = [], [], []
    for _, r in out.iterrows():
        clicks = int(r["clicks"])
        conv = int(round(r["conversions"]))
        rest_conv = int(round(total_conv - conv))
        rest_clicks = int(total_clicks - clicks)
        p, eff, direction = two_proportion_test(conv, clicks, rest_conv, rest_clicks)
        lo, hi = wilson_interval(conv, clicks, cfg.confidence)
        p_values.append(p)
        effects.append(eff)
        directions.append(direction)
        wlo.append(lo)
        whi.append(hi)
        cvrs.append(conv / clicks if clicks > 0 else 0.0)

    out["cvr"] = cvrs
    out["p_value"] = p_values
    out["effect"] = effects
    out["direction"] = directions
    out["wilson_low"] = wlo
    out["wilson_high"] = whi

    recs, reasons = [], []
    for _, r in out.iterrows():
        clicks = int(r["clicks"])
        conv = int(round(r["conversions"]))
        cost = float(r["cost"])
        p = float(r["p_value"])
        direction = r["direction"]
        shrunk = float(r["shrunk_cvr"])
        sig = p < alpha

        if clicks < cfg.min_clicks:
            if cost >= cfg.negative_min_cost and conv == 0:
                rec, reason = "HOLD-INCUBATING", f"spend €{cost:.0f}, {clicks} clicks - watch for negative"
            else:
                rec, reason = "LOW-VOLUME", f"only {clicks} clicks - shrunk CVR {shrunk:.1%}"
        elif conv == 0 and cost >= cfg.negative_min_cost and sig and direction == "below":
            rec, reason = "NEGATIVE", f"€{cost:.0f} spend, 0 deposits, p={p:.3f}"
        elif sig and direction == "below":
            rec, reason = "PRUNE", f"CVR {r['cvr']:.1%} < group {r['group_mean_cvr']:.1%}, p={p:.3f}"
        elif sig and direction == "above" and shrunk >= r["group_mean_cvr"] + cfg.promote_min_shrunk_uplift:
            rec, reason = "PROMOTE", f"CVR {r['cvr']:.1%} > group, shrunk {shrunk:.1%}, p={p:.3f}"
        elif clicks < cfg.incubation_clicks:
            rec, reason = "HOLD-INCUBATING", f"{clicks} clicks, not yet significant (p={p:.3f})"
        else:
            rec, reason = "HOLD-INCUBATING", f"no significant signal (p={p:.3f})"
        recs.append(rec)
        reasons.append(reason)

    out["recommendation"] = recs
    out["reason"] = reasons
    return out
