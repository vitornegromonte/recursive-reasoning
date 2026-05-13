"""
Statistical utilities for MI experiments.

Provides bootstrap confidence intervals, paired t-tests, effect sizes,
and significance testing infrastructure.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    samples: list[float],
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    ci: float = 95.0,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Bootstrap confidence interval for a statistic over 1-D samples.

    Args:
        samples: List of sample values.
        statistic: Function to compute statistic (default: np.mean).
        n_bootstrap: Number of bootstrap iterations.
        ci: Confidence interval percentage (default: 95.0).
        rng: Random number generator (default: np.random.default_rng()).

    Returns:
        Dict with mean, std, ci_low, ci_high, n, n_bootstrap.
    """
    if rng is None:
        rng = np.random.default_rng()

    arr = np.array(samples, dtype=float)
    if len(arr) == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "n": 0,
            "n_bootstrap": n_bootstrap,
        }

    boot = np.array([
        statistic(rng.choice(arr, size=len(arr), replace=True))
        for _ in range(n_bootstrap)
    ])

    lo = (100.0 - ci) / 2.0
    hi = 100.0 - lo
    ddof = 1 if len(arr) >= 2 else 0

    return {
        "mean": round(float(statistic(arr)), 6),
        "std": round(float(np.std(arr, ddof=ddof)), 6),
        "ci_low": round(float(np.percentile(boot, lo)), 6),
        "ci_high": round(float(np.percentile(boot, hi)), 6),
        "n": len(arr),
        "n_bootstrap": n_bootstrap,
    }


def paired_t_test(
    group_a: list[float],
    group_b: list[float],
    alternative: str = "two-sided",
) -> dict[str, float]:
    """
    Paired t-test for comparing two related groups.

    Args:
        group_a: First group of values.
        group_b: Second group of values (must have same length as group_a).
        alternative: 'two-sided', 'greater', or 'less'.

    Returns:
        Dict with t_stat, p_value, df, mean_diff.
    """
    if len(group_a) != len(group_b):
        raise ValueError("Groups must have equal length for paired t-test")

    if len(group_a) < 2:
        return {"t_stat": float("nan"), "p_value": float("nan"), "df": 0, "mean_diff": float("nan")}

    arr_a = np.array(group_a, dtype=float)
    arr_b = np.array(group_b, dtype=float)
    diff = arr_a - arr_b

    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    n = len(diff)

    if std_diff == 0:
        return {"t_stat": 0.0, "p_value": 1.0, "df": n - 1, "mean_diff": mean_diff}

    t_stat = mean_diff / (std_diff / np.sqrt(n))
    df = n - 1

    from scipy import stats
    p_value = stats.t.sf(np.abs(t_stat), df)

    if alternative == "two-sided":
        p_value = 2 * p_value
    elif alternative == "greater":
        p_value = stats.t.sf(t_stat, df)
    elif alternative == "less":
        p_value = stats.t.cdf(t_stat, df)

    return {
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "df": int(df),
        "mean_diff": round(float(mean_diff), 6),
    }


def cohens_d(
    group_a: list[float],
    group_b: list[float],
) -> float:
    """
    Cohen's d effect size between two groups.

    Args:
        group_a: First group of values.
        group_b: Second group of values.

    Returns:
        Cohen's d (positive means group_a > group_b).
    """
    arr_a = np.array(group_a, dtype=float)
    arr_b = np.array(group_b, dtype=float)

    mean_a = np.mean(arr_a)
    mean_b = np.mean(arr_b)

    var_a = np.var(arr_a, ddof=1)
    var_b = np.var(arr_b, ddof=1)
    n_a = len(arr_a)
    n_b = len(arr_b)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float((mean_a - mean_b) / pooled_std)


def permutation_test(
    group_a: list[float],
    group_b: list[float],
    n_permutations: int = 10_000,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """
    Permutation test for difference in means.

    Args:
        group_a: First group of values.
        group_b: Second group of values.
        n_permutations: Number of permutations.
        rng: Random number generator.

    Returns:
        Dict with observed_diff, p_value.
    """
    if rng is None:
        rng = np.random.default_rng()

    arr_a = np.array(group_a, dtype=float)
    arr_b = np.array(group_b, dtype=float)

    observed_diff = np.mean(arr_a) - np.mean(arr_b)

    combined = np.concatenate([arr_a, arr_b])
    n_a = len(arr_a)

    count = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = combined[:n_a]
        perm_b = combined[n_a:]
        perm_diff = np.mean(perm_a) - np.mean(perm_b)
        if np.abs(perm_diff) >= np.abs(observed_diff):
            count += 1

    p_value = (count + 1) / (n_permutations + 1)

    return {
        "observed_diff": round(float(observed_diff), 6),
        "p_value": round(float(p_value), 6),
        "n_permutations": n_permutations,
    }


def significance_threshold(
    p_value: float,
    alpha: float = 0.05,
) -> str:
    """
    Determine significance level string.

    Args:
        p_value: The p-value to test.
        alpha: Significance threshold (default: 0.05).

    Returns:
        String indicating significance level.
    """
    if p_value >= alpha:
        return "ns"
    elif p_value >= 0.01:
        return "*"
    elif p_value >= 0.001:
        return "**"
    else:
        return "***"


def format_result_with_significance(
    mean: float,
    std: float,
    p_value: float | None = None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    alpha: float = 0.05,
) -> str:
    """
    Format a result string with optional significance.

    Args:
        mean: Mean value.
        std: Standard deviation.
        p_value: Optional p-value for significance.
        ci_low: Optional lower confidence interval.
        ci_high: Optional upper confidence interval.
        alpha: Significance threshold.

    Returns:
        Formatted string (e.g., "0.45 ± 0.02 (*)").
    """
    base = f"{mean:.3f} ± {std:.3f}"

    if ci_low is not None and ci_high is not None:
        base = f"{mean:.3f} [{ci_low:.3f}, {ci_high:.3f}]"

    if p_value is not None:
        sig = significance_threshold(p_value, alpha)
        if sig != "ns":
            return f"{base} ({sig})"

    return base