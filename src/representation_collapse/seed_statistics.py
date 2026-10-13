"""Prospective seed-level behavior reduction; no protocol or claim acceptance.

The family and margins below implement the *proposal* dated 2026-10-12. They
are not a frozen analysis plan. Student-t reports use an unverified Gaussian
approximation for the eligibility-conditional seed population. Missing or
ineligible seeds cannot be dropped, and all eight family slots are retained.
This module never reads artifacts, chooses checkpoints, or evaluates models.
"""

from __future__ import annotations

import hashlib
import json
import math
from numbers import Integral, Real
import statistics

import numpy as np


CELL_ORDER = (("gru", "categorical"), ("gru", "continuous"),
              ("lstm", "categorical"), ("lstm", "continuous"))
EFFECT_ORDER = ("delta16", "DID")
FAMILY_ORDER = tuple(f"{model}_{task}_{effect}"
                     for model, task in CELL_ORDER for effect in EFFECT_ORDER)
ALPHA = 0.05
SEI = 0.02
CONTINUOUS_POPULATION_VARIANCE = 1.0 / 3.0
MIN_SUPPORTED_T_QUANTILE_TAIL = 1e-100
PROPOSED_CONFIRMATION_SEEDS = tuple(range(3, 13))
LOSS_UNITS = {"categorical": "error_indicator", "continuous": "raw_squared_error"}
_ASSUMPTIONS = (
    "independent_training_seeds_in_validation_eligibility_conditional_population",
    "Gaussian_seed_effect_distribution_for_exact_t_calibration",
    "otherwise_unverified_approximate_t_calibration_no_FWER_guarantee",
)


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _beta_fraction(a, b, x):
    """Modified-Lentz evaluation of the incomplete-beta continued fraction."""
    tiny = 1e-300

    def protect(value):
        return math.copysign(tiny, value) if abs(value) < tiny else value

    c = 1.0
    d = 1.0 / protect(1.0 - (a + b) * x / (a + 1.0))
    result = d
    for m in range(1, 10001):
        m2 = 2 * m
        even = m * (b - m) * x / ((a - 1.0 + m2) * (a + m2))
        d = 1.0 / protect(1.0 + even * d)
        c = protect(1.0 + even / c)
        result *= d * c
        odd = -(a + m) * (a + b + m) * x / ((a + m2) * (a + 1.0 + m2))
        d = 1.0 / protect(1.0 + odd * d)
        c = protect(1.0 + odd / c)
        factor = d * c
        result *= factor
        if abs(factor - 1.0) <= 8.0 * np.finfo(float).eps:
            if not math.isfinite(result) or result <= 0:
                raise ArithmeticError("invalid incomplete-beta continued fraction")
            return result
    raise ArithmeticError("incomplete-beta continued fraction did not converge")


def _regularized_beta(x, a, b):
    if x == 0:
        return 0.0
    if x == 1:
        return 1.0
    weight = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        value = weight * _beta_fraction(a, b, x) / a
    else:
        value = 1.0 - weight * _beta_fraction(b, a, 1.0 - x) / b
    if not math.isfinite(value) or not -1e-14 <= value <= 1.0 + 1e-14:
        raise ArithmeticError("invalid regularized incomplete beta")
    return min(1.0, max(0.0, value))


def student_t_survival(statistic, degrees_of_freedom):
    """Numerical upper-tail Student-t probability, without SciPy dependency.

    For positive t, SF(t) = I_[df/(df+t²)](df/2, 1/2)/2. This distribution
    calculation is distinct from whether a seed-effect test has valid coverage.
    Numerical tails can underflow at extreme arguments; family reduction marks
    such a p-value unavailable instead of accepting a silent zero.
    """
    t = _finite(statistic, "t statistic")
    df = _positive_integer(degrees_of_freedom, "degrees of freedom")
    if t == 0:
        return 0.5
    root_df = math.sqrt(df)
    if abs(t) < 1.0:
        # Near zero, df/(df+t²) rounds to one and erases the CDF slope.
        # Evaluate the complementary small beta argument instead. Restrict
        # this subtraction to genuinely central arguments: |t|<sqrt(df)
        # includes extreme tails at high df, where 0.5-central_mass cancels.
        central_x = (t / math.hypot(t, root_df)) ** 2
        central_mass = 0.5 * _regularized_beta(central_x, 0.5, df / 2.0)
        return 0.5 - central_mass if t > 0 else 0.5 + central_mass
    x = (root_df / math.hypot(t, root_df)) ** 2
    positive_tail = 0.5 * _regularized_beta(x, df / 2.0, 0.5)
    return positive_tail if t > 0 else 1.0 - positive_tail


def student_t_upper_quantile(degrees_of_freedom, tail_probability):
    """Positive t quantile for upper tails in [1e-100, 1/2).

    More extreme tails are unsupported: for df=1 the squared beta argument
    eventually underflows, creating a false inversion boundary. Reject them
    explicitly rather than returning that boundary as a finite quantile. The
    supported range and a relative survival-probability residual are checked.
    """
    df = _positive_integer(degrees_of_freedom, "degrees of freedom")
    probability = _finite(tail_probability, "upper-tail probability")
    if not 0 < probability < 0.5:
        raise ValueError("upper-tail probability must be strictly between zero and 1/2")
    if probability < MIN_SUPPORTED_T_QUANTILE_TAIL:
        raise ArithmeticError("Student-t quantile tail below supported numerical limit (1e-100)")
    lower, upper = 0.0, 1.0
    for _ in range(1024):
        upper_tail = student_t_survival(upper, df)
        if upper_tail == 0:
            raise ArithmeticError("Student-t quantile bracket reached survival underflow")
        if upper_tail <= probability:
            break
        upper *= 2.0
        if not math.isfinite(upper):
            raise ArithmeticError("Student-t quantile cannot be bracketed")
    else:
        raise ArithmeticError("Student-t quantile cannot be bracketed")
    for _ in range(160):
        midpoint = lower + (upper - lower) / 2.0
        if midpoint == lower or midpoint == upper:
            break
        midpoint_tail = student_t_survival(midpoint, df)
        if midpoint_tail == 0:
            raise ArithmeticError("Student-t quantile bisection reached survival underflow")
        if midpoint_tail > probability:
            lower = midpoint
        else:
            upper = midpoint
    result = lower + (upper - lower) / 2.0
    residual = abs(student_t_survival(result, df) / probability - 1.0)
    if not math.isfinite(residual) or residual > 2e-11:
        raise ArithmeticError("Student-t quantile failed relative survival-residual check")
    return result


def paired_behavior_effects(periodic_losses, control_losses, *, task, input_units,
                            periodic_pair_ids, control_pair_ids,
                            episodes_per_coordinate=128):
    """Reduce paired R2/R16 episode losses to one training seed's effects.

    Each argument is an R2/R16 mapping of arrays in [query coordinate 0..7,
    episode] order. Categorical entries are error indicators; continuous entries
    are raw squared errors, normalized by the fixed population variance 1/3.
    Pair IDs must agree elementwise between arms and be unique within a horizon.
    R2/R16 need not share episode IDs: DID pairs the two effects within a seed,
    without treating episodes or coordinates as training replicates.
    """
    if task not in LOSS_UNITS or input_units != LOSS_UNITS[task]:
        raise ValueError("task and raw input loss units must match")
    count = _positive_integer(episodes_per_coordinate, "episodes per coordinate")
    mappings = (periodic_losses, control_losses, periodic_pair_ids, control_pair_ids)
    if any(not isinstance(value, dict) or set(value) != {"R2", "R16"} for value in mappings):
        raise ValueError("exactly R2/R16 loss and pair-ID mappings required")
    arm_means, paired_raw_differences, paired_deltas, pair_hashes = {}, {}, {}, {}
    for horizon in ("R2", "R16"):
        losses = []
        ids = []
        for mapping in (periodic_losses, control_losses):
            array = np.asarray(mapping[horizon])
            if array.dtype.kind not in "biuf" or array.shape != (8, count):
                raise ValueError("loss arrays require eight coordinates and declared numeric episodes")
            array = array.astype(np.float64)
            if not np.all(np.isfinite(array)) or np.any(array < 0):
                raise ValueError("loss arrays must be finite and nonnegative")
            if task == "categorical" and not np.all((array == 0) | (array == 1)):
                raise ValueError("categorical episode losses must be binary error indicators")
            losses.append(array)
        for mapping in (periodic_pair_ids, control_pair_ids):
            array = np.asarray(mapping[horizon], dtype=object)
            if array.shape != (8, count):
                raise ValueError("pair-ID arrays must match coordinate/episode shape")
            flat = array.ravel().tolist()
            if any(not isinstance(value, str) or not value.strip() for value in flat) or len(set(flat)) != 8 * count:
                raise ValueError("nonempty unique pair IDs required within each horizon")
            ids.append(array)
        if not np.array_equal(ids[0], ids[1]):
            raise ValueError("periodic/control pair IDs must agree elementwise")
        periodic, control = losses
        # Equal coordinate weights are explicit even though fixed strata sizes
        # also allow a flat mean. Within-arm means remain in raw loss units.
        mean = lambda rows: statistics.mean(statistics.mean(row.tolist()) for row in rows)
        arm_means[horizon] = {"periodic": mean(periodic), "matched_control": mean(control)}
        scale = CONTINUOUS_POPULATION_VARIANCE if task == "continuous" else 1.0
        with np.errstate(over="raise", invalid="raise"):
            try:
                paired_raw_differences[horizon] = mean(periodic - control)
                paired_deltas[horizon] = paired_raw_differences[horizon] / scale
            except FloatingPointError as error:
                raise ValueError("paired loss normalization overflowed") from error
        pair_hashes[horizon] = hashlib.sha256(json.dumps(ids[0].tolist(), separators=(",", ":"),
                                                       ensure_ascii=False).encode()).hexdigest()
    effects = {"delta2": paired_deltas["R2"], "delta16": paired_deltas["R16"],
               "DID": paired_deltas["R16"] - paired_deltas["R2"]}
    if not all(math.isfinite(value) for value in effects.values()):
        raise ValueError("derived seed effect is not finite")
    return {"loss_units": input_units, "arm_mean_losses": arm_means,
            "paired_mean_raw_loss_differences": paired_raw_differences, "effects": effects,
            "effect_units": "error_probability" if task == "categorical" else "population_normalized_MSE",
            "continuous_population_variance": CONTINUOUS_POPULATION_VARIANCE if task == "continuous" else None,
            "coordinates": list(range(8)), "episode_pairs_per_coordinate": count,
            "episode_pairs_per_horizon": 8 * count, "pair_id_sha256": pair_hashes,
            "within_horizon_episode_pairing_checked": True, "cross_horizon_episode_pairing_assumed": False,
            "matches_proposed_episode_count": count == 128, "training_seed_replications": 1,
            "scientific_acceptance": False}


def seed_t_summary(seed_effects):
    """Mean, sample SD, approximate t intervals and one-sided boundary p-value.

    This standalone descriptive calculation does not apply Holm. Zero sample
    SD, an unrepresentable statistic, or a tail underflow yields no p-value or
    interval and an explicit reason; the finite descriptive mean is retained.
    """
    if not isinstance(seed_effects, (list, tuple)) or len(seed_effects) < 2:
        raise ValueError("at least two declared independent seed effects required")
    values = [_finite(value, "seed effect") for value in seed_effects]
    mean = float(statistics.mean(values))
    result = {"n_training_seeds": len(values), "mean": mean, "sample_sd": None,
              "sample_variance_denominator": len(values) - 1, "standard_error": None,
              "t_statistic_at_SEI": None, "raw_approximate_p": None,
              "ordinary_95_t_interval": None, "bonferroni_99_375_t_interval": None,
              "numerically_available": False, "unavailable_reason": None,
              "method": "one_sample_t_on_paired_seed_effects_approximate_unless_Gaussian",
              "SEI": SEI, "assumptions": list(_ASSUMPTIONS),
              "calibration_assumptions_verified": False, "finite_sample_exactness_claimed": False}
    try:
        sd = float(statistics.stdev(values))
        if not math.isfinite(sd):
            raise ArithmeticError("nonfinite_seed_sd")
        result["sample_sd"] = sd
        se = sd / math.sqrt(len(values))
        result["standard_error"] = se
        if se == 0:
            result["unavailable_reason"] = "zero_seed_sd_or_underflowed_standard_error"
            return result
        statistic = (mean - SEI) / se
        if not math.isfinite(statistic):
            raise ArithmeticError("nonfinite_t_statistic")
        p_value = student_t_survival(statistic, len(values) - 1)
        if p_value == 0:
            raise ArithmeticError("t_upper_tail_underflow")
        intervals = {}
        for key, tail in (("ordinary_95_t_interval", ALPHA / 2),
                          ("bonferroni_99_375_t_interval", ALPHA / (2 * len(FAMILY_ORDER)))):
            critical = student_t_upper_quantile(len(values) - 1, tail)
            endpoints = [mean - critical * se, mean + critical * se]
            if not all(math.isfinite(value) for value in endpoints):
                raise ArithmeticError("nonfinite_t_interval")
            intervals[key] = {"lower": endpoints[0], "upper": endpoints[1],
                              "critical_t": critical, "degrees_of_freedom": len(values) - 1,
                              "coverage": 1 - 2 * tail, "coverage_is_approximate": True,
                              "simultaneous_coverage_guarantee_without_assumptions": False}
        result.update(intervals)
        result.update(t_statistic_at_SEI=statistic, raw_approximate_p=p_value,
                      numerically_available=True)
    except (ArithmeticError, OverflowError) as error:
        result["unavailable_reason"] = str(error)
    return result


def _outcome_effects(task, outcome):
    if not isinstance(outcome, dict) or outcome.get("loss_units") != LOSS_UNITS[task]:
        raise ValueError("outcome requires task-matched raw loss units")
    means = outcome.get("arm_mean_losses")
    if not isinstance(means, dict) or set(means) != {"R2", "R16"}:
        raise ValueError("outcome requires exactly R2/R16 arm-mean losses")
    paired = outcome.get("paired_mean_raw_loss_differences")
    if not isinstance(paired, dict) or set(paired) != {"R2", "R16"}:
        raise ValueError("outcome requires R2/R16 means of paired raw loss differences")
    deltas = {}
    scale = CONTINUOUS_POPULATION_VARIANCE if task == "continuous" else 1.0
    for horizon, arms in means.items():
        if not isinstance(arms, dict) or set(arms) != {"periodic", "matched_control"}:
            raise ValueError("outcome requires both periodic and matched_control means")
        values = [_finite(arms[arm], "arm mean loss") for arm in ("periodic", "matched_control")]
        if min(values) < 0 or (task == "categorical" and max(values) > 1):
            raise ValueError("arm mean loss outside task support")
        difference = _finite(paired[horizon], "paired mean raw loss difference")
        # Averages of differences preserve small paired contrasts when arm
        # means are large. Check arm-mean consistency within floating rounding
        # resolution, then retain the explicitly paired estimator. This check
        # is not a substitute for independent raw-array/hash verification.
        tolerance = 8 * max(math.ulp(value) for value in (*values, difference))
        if abs(difference - (values[0] - values[1])) > tolerance:
            raise ValueError("paired loss difference disagrees with arm means beyond rounding resolution")
        deltas[horizon] = difference / scale
    effects = {"delta2": deltas["R2"], "delta16": deltas["R16"],
               "DID": deltas["R16"] - deltas["R2"]}
    if not all(math.isfinite(value) for value in effects.values()):
        raise ValueError("arm mean normalization produced a nonfinite seed effect")
    if "effects" in outcome:
        supplied = outcome["effects"]
        if not isinstance(supplied, dict) or set(supplied) != set(effects):
            raise ValueError("derived effect keys disagree with arm means")
        for name, expected in effects.items():
            value = _finite(supplied[name], "supplied seed effect")
            if not math.isclose(value, expected, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("derived seed effect disagrees with arm means")
    return effects


def reduce_seed_family(records, *, declared_seeds):
    """Reduce all four cells and eight slots without dropping failed seeds.

    Records require model/task/trained_seed, a boolean validation_eligible,
    optional unavailable_reason, and (when available) an outcome packet with
    loss_units, R2/R16 arm_mean_losses and paired_mean_raw_loss_differences.
    IDs outside the caller's prospectively
    declared seed list, duplicate records and malformed outcomes raise errors.
    Known missing/ineligible/numerically unavailable records are explicit ledger
    entries. Any such seed makes its cell inconclusive with both decision_p=1.

    Holm outputs are *approximate candidate calculations*, never an accepted
    harm decision. The original family stays eight even if only one cell has
    outcomes. Artifact verification, eligibility verification, bootstrap reports
    and protocol acceptance are intentionally outside this pure reducer.
    """
    if not isinstance(declared_seeds, (list, tuple)) or len(declared_seeds) < 2:
        raise ValueError("caller must declare at least two independent training seeds")
    seeds = []
    for seed in declared_seeds:
        if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
            raise ValueError("declared seed IDs must be nonnegative integers")
        seeds.append(int(seed))
    if len(set(seeds)) != len(seeds):
        raise ValueError("declared seed IDs must be unique")
    if not isinstance(records, (list, tuple)) or any(not isinstance(record, dict) for record in records):
        raise ValueError("seed records must be a dictionary sequence")
    indexed = {}
    for record in records:
        cell = (record.get("model"), record.get("task"))
        seed = record.get("trained_seed")
        if cell not in CELL_ORDER or isinstance(seed, bool) or not isinstance(seed, Integral) or seed not in seeds:
            raise ValueError("record identity is outside declared cells/seeds")
        key = (cell, int(seed))
        if key in indexed:
            raise ValueError("duplicate seed record; replacement/rescue is not allowed")
        if not isinstance(record.get("validation_eligible"), bool):
            raise ValueError("explicit boolean validation eligibility required")
        reason = record.get("unavailable_reason")
        if reason is not None and (not isinstance(reason, str) or not reason.strip()):
            raise ValueError("unavailable reason must be a nonempty string or null")
        if not record["validation_eligible"] and reason is None:
            raise ValueError("ineligible seed requires an explicit unavailable reason")
        observed = _outcome_effects(cell[1], record["outcome"]) if record.get("outcome") is not None else None
        effects = None
        if record["validation_eligible"] and reason is None:
            if observed is None:
                reason = "missing_outcome_after_validation_eligibility"
            else:
                effects = observed
        indexed[key] = {"trained_seed": int(seed), "validation_eligible": record["validation_eligible"],
                        "available": effects is not None, "unavailable_reason": reason,
                        "effects": effects, "observed_outcome_effects": observed}

    cells, slots = [], []
    for model, task in CELL_ORDER:
        ledger = [indexed.get(((model, task), seed),
                             {"trained_seed": seed, "validation_eligible": None, "available": False,
                              "unavailable_reason": "missing_declared_seed_record", "effects": None,
                              "observed_outcome_effects": None})
                  for seed in seeds]
        complete = all(row["available"] for row in ledger)
        cells.append({"model": model, "task": task, "seed_ledger": ledger,
                      "all_declared_seeds_available": complete,
                      "available_training_seeds": sum(row["available"] for row in ledger),
                      "primary_available_case_mean_reported": False})
        for effect in EFFECT_ORDER:
            summary = seed_t_summary([row["effects"][effect] for row in ledger]) if complete else None
            available = complete and summary["numerically_available"]
            slots.append({"slot": f"{model}_{task}_{effect}", "model": model, "task": task,
                          "effect": effect, "complete_declared_seed_set": complete,
                          "t_summary": summary, "decision_p": summary["raw_approximate_p"] if available else 1.0,
                          "primary_numerically_available": available,
                          "unavailable_reason": None if available else
                          ("incomplete_or_ineligible_declared_seed_set" if not complete else summary["unavailable_reason"]),
                          "scientific_decision": "pending_protocol_acceptance" if available else "inconclusive"})

    order = sorted(range(len(slots)), key=lambda index: (slots[index]["decision_p"], index))
    running_adjusted, stopped = 0.0, False
    for rank, index in enumerate(order):
        slot = slots[index]
        factor = len(slots) - rank
        running_adjusted = max(running_adjusted, factor * slot["decision_p"])
        threshold = ALPHA / factor
        reject = not stopped and slot["decision_p"] <= threshold and slot["primary_numerically_available"]
        if not reject:
            stopped = True
        slot.update(holm_rank=rank + 1, holm_threshold=threshold,
                    approximate_holm_adjusted_p=min(1.0, running_adjusted),
                    approximate_holm_rejects=reject, accepted_harm_claim=False)
    return {"schema_version": 1, "protocol_status": "unfrozen_statistics_v2_proposal",
            "declared_training_seeds": seeds, "replication_unit": "independent_training_seed",
            "matches_proposed_confirmation_seed_set": tuple(seeds) == PROPOSED_CONFIRMATION_SEEDS,
            "family_order": list(FAMILY_ORDER), "family_size": len(FAMILY_ORDER),
            "alpha": ALPHA, "SEI": SEI, "cells": cells, "slots": slots,
            "assumptions": list(_ASSUMPTIONS), "calibration_assumptions_verified": False,
            "scientific_acceptance": False, "protocol_frozen": False,
            "bootstrap_interval_implemented": False, "both_required_reporting_intervals_verified": False,
            "eligibility_independently_verified": False, "artifact_hashes_independently_verified": False,
            "equivalence_enabled": False, "sign_flip_performed": False,
            "accepted_harm_claim": False, "compute_or_model_calls_performed": False}
