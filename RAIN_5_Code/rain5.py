from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import datetime as dt
import itertools
import math
import time

from .utils import read_json, write_json, find_latest, copy_inputs_to_output, now_stamp, stable_id, normalise_text
from .profile_loader import load_profiles
from .calibration import load_labelled_matches, pair_bvh_key, pair_key
from .hvti_estimator import (
    estimate_directional_fit,
    score_hvti,
    estimate_component_scores,
    final_ranking_score,
)
from .roi_estimator import estimate_roi_usd
from .supervised_ranker import (
    candidate_training_row,
    train_precision50_models,
    apply_supervised_scores,
)
from .prototype_ranker import (
    build_pair_prototype_model,
    serialise_pair_prototype_model,
    apply_prototype_antigeneric_scores,
    apply_person_saturation_penalty_to_suggestions,
    candidate_rank_row,
    build_label_coverage_report,
    write_label_coverage_report,
)
from .bvh_precision_ranker import (
    train_bvh_precision_models,
    apply_bvh_precision_scores,
    select_bvh_candidates_for_pair,
    bvh_precision_model_summary,
)
from .bvh_recall_balancer import (
    DEFAULT_FLOOR_SPEC,
    DEFAULT_BOOST_SPEC,
    parse_bvh_floor_spec,
    parse_bvh_boost_spec,
    apply_bvh_recall_boost,
    balance_bvh_type_floors,
    count_by_bvh,
)


def build_bvh_id(person_a_id: str, person_b_id: str, bvh_type_id: str) -> str:
    return stable_id("BVH", f"{person_a_id}|{person_b_id}|{bvh_type_id}", width=10)


def format_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"


def format_eta(now_ts: float, seconds_remaining: float) -> str:
    eta = dt.datetime.fromtimestamp(now_ts + max(0, seconds_remaining))
    return eta.strftime("%H:%M:%S")


def make_logger(log_path: Path):
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(message: str) -> None:
        timestamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{timestamp}] {message}"
        print(line, flush=True)
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    return log


def apply_profile_sample(profiles: List[Dict[str, Any]], profile_sample_pct: float, log) -> List[Dict[str, Any]]:
    if profile_sample_pct <= 0:
        raise ValueError("--profile-sample-pct must be greater than 0.")
    if profile_sample_pct >= 100:
        log(f"[SAMPLE] Using 100% of profiles: {len(profiles)}")
        return profiles

    original_count = len(profiles)
    sample_count = max(2, int(math.ceil(original_count * (profile_sample_pct / 100.0))))
    sampled = profiles[:sample_count]
    log(f"[SAMPLE] Using {profile_sample_pct:.2f}% of profiles: {sample_count} of {original_count}")
    return sampled


def estimate_total_work(profile_count: int, bvh_type_count: int, max_pairs: Optional[int]) -> Dict[str, int]:
    total_pairs_all = profile_count * (profile_count - 1) // 2
    total_pairs = min(total_pairs_all, max_pairs) if max_pairs is not None else total_pairs_all
    total_bvh_evaluations = total_pairs * bvh_type_count
    return {
        "total_pairs_all": total_pairs_all,
        "total_pairs": total_pairs,
        "total_bvh_evaluations": total_bvh_evaluations,
    }


def maybe_log_progress(
    *,
    log,
    processed_evaluations: int,
    total_evaluations: int,
    candidate_count: int,
    suggestion_count_so_far: int,
    start_ts: float,
    last_progress_ts: float,
    progress_every: int,
    progress_seconds: float,
) -> float:
    now = time.time()
    should_log = False

    if processed_evaluations == 1:
        should_log = True
    elif progress_every > 0 and processed_evaluations % progress_every == 0:
        should_log = True
    elif progress_seconds > 0 and (now - last_progress_ts) >= progress_seconds:
        should_log = True
    elif processed_evaluations == total_evaluations:
        should_log = True

    if not should_log:
        return last_progress_ts

    elapsed = now - start_ts
    rate = processed_evaluations / elapsed if elapsed > 0 else 0.0
    remaining = max(0, total_evaluations - processed_evaluations)
    remaining_seconds = remaining / rate if rate > 0 else 0.0
    pct = (processed_evaluations / total_evaluations * 100.0) if total_evaluations else 100.0

    log(
        "[PROGRESS] "
        f"{processed_evaluations:,}/{total_evaluations:,} BVH evaluations "
        f"({pct:.2f}%) | elapsed {format_duration(elapsed)} | "
        f"rate {rate:,.1f}/sec | ETA {format_eta(now, remaining_seconds)} "
        f"({format_duration(remaining_seconds)} remaining) | "
        f"candidates {candidate_count:,} | suggestions-kept {suggestion_count_so_far:,}"
    )
    return now


def normalise_weights(hvti_structure: Dict[str, Any]) -> None:
    total = sum(float(p.get("DefaultWeight", 0.0)) for p in hvti_structure.get("parameters", [])) or 1.0
    for p in hvti_structure.get("parameters", []):
        p["DefaultWeight"] = round(float(p.get("DefaultWeight", 0.0)) / total, 6)


def profile_by_name(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {normalise_text(p.get("full_name", "")): p for p in profiles if p.get("full_name")}


def bvh_by_id(bvh_types: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(b.get("BVHTypeID", "")).upper(): b for b in bvh_types}


def average_params(param_rows: List[Dict[str, float]], parameter_ids: List[str]) -> Dict[str, float]:
    if not param_rows:
        return {pid: 0.0 for pid in parameter_ids}
    return {
        pid: sum(float(row.get(pid, 0.0)) for row in param_rows) / len(param_rows)
        for pid in parameter_ids
    }


def tune_hvti_weights_from_labels(
    profiles: List[Dict[str, Any]],
    bvh_types: List[Dict[str, Any]],
    hvti_structure: Dict[str, Any],
    calibration: Dict[str, Any],
    negative_samples_per_positive: int,
    log,
) -> Dict[str, Any]:
    """
    Supervised but lightweight tuning using matchesCreated.csv as positive labels.

    No external ML libraries are used. We compare HVTI parameter averages for:
    - positive labelled pair+BVH examples
    - deterministic negative pair+BVH examples
    and increase weights for parameters that separate positives from negatives.
    """
    positive_examples = calibration.get("positive_examples", [])
    parameter_ids = [p["ParameterID"] for p in hvti_structure.get("parameters", [])]
    if not positive_examples or not parameter_ids:
        return {
            "enabled": False,
            "reason": "No positive examples or HVTI parameters available.",
            "label_count": len(positive_examples),
        }

    p_by_name = profile_by_name(profiles)
    b_by_id = bvh_by_id(bvh_types)

    positive_param_rows: List[Dict[str, float]] = []
    used_positive_count = 0

    for ex in positive_examples:
        pa = p_by_name.get(normalise_text(ex.get("PersonAName", "")))
        pb = p_by_name.get(normalise_text(ex.get("PersonBName", "")))
        bvh = b_by_id.get(str(ex.get("BVHTypeID", "")).upper())
        if not pa or not pb or not bvh:
            continue

        fit_ab = estimate_directional_fit(pa, pb, bvh)
        fit_ba = estimate_directional_fit(pb, pa, bvh)

        # Use the stronger direction as the labelled signal.
        temp_cal = {"label_count": calibration.get("label_count", 0), "bvh_type_priors": {}, "bvh_type_avg_score": {}, "bvh_type_avg_value_gbp": {}}
        score_ab = score_hvti(fit_ab["parameters"], hvti_structure, temp_cal, bvh["BVHTypeID"])
        score_ba = score_hvti(fit_ba["parameters"], hvti_structure, temp_cal, bvh["BVHTypeID"])
        selected = fit_ba if score_ba["EstimatedHVTIScore"] > score_ab["EstimatedHVTIScore"] else fit_ab
        positive_param_rows.append(selected["parameters"])
        used_positive_count += 1

    positive_pair_bvh_names = calibration.get("positive_pair_bvh_names", set())
    negative_param_rows: List[Dict[str, float]] = []
    negative_target = max(10, used_positive_count * max(1, negative_samples_per_positive))

    # Deterministic negative sampling from profile pairs and BVH types.
    for i, pa in enumerate(profiles):
        if len(negative_param_rows) >= negative_target:
            break
        for pb in profiles[i + 1:]:
            if len(negative_param_rows) >= negative_target:
                break
            for bvh in bvh_types:
                if len(negative_param_rows) >= negative_target:
                    break
                key = pair_bvh_key(pa.get("full_name", ""), pb.get("full_name", ""), bvh.get("BVHTypeID", ""))
                if key in positive_pair_bvh_names:
                    continue
                fit = estimate_directional_fit(pa, pb, bvh)
                negative_param_rows.append(fit["parameters"])

    if not positive_param_rows or not negative_param_rows:
        return {
            "enabled": False,
            "reason": "Insufficient positive/negative examples for tuning.",
            "positive_examples_used": len(positive_param_rows),
            "negative_examples_used": len(negative_param_rows),
        }

    pos_avg = average_params(positive_param_rows, parameter_ids)
    neg_avg = average_params(negative_param_rows, parameter_ids)

    old_weights = {p["ParameterID"]: float(p.get("DefaultWeight", 0.0)) for p in hvti_structure.get("parameters", [])}
    separations = {}
    for pid in parameter_ids:
        separations[pid] = max(0.001, pos_avg.get(pid, 0.0) - neg_avg.get(pid, 0.0))

    sep_total = sum(separations.values()) or 1.0
    learned_weights = {pid: separations[pid] / sep_total for pid in parameter_ids}

    # Blend rather than fully replace: keeps domain priors stable and uses labels to tune.
    blend = 0.55
    new_weights = {}
    for pid in parameter_ids:
        new_weights[pid] = (old_weights.get(pid, 0.0) * (1 - blend)) + (learned_weights.get(pid, 0.0) * blend)

    total_new = sum(new_weights.values()) or 1.0
    new_weights = {pid: round(w / total_new, 6) for pid, w in new_weights.items()}

    for p in hvti_structure.get("parameters", []):
        pid = p["ParameterID"]
        p["DefaultWeight"] = new_weights.get(pid, p.get("DefaultWeight", 0.0))

    normalise_weights(hvti_structure)

    tuning_result = {
        "enabled": True,
        "method": "positive_vs_negative_parameter_separation_blended_weights",
        "positive_examples_used": len(positive_param_rows),
        "negative_examples_used": len(negative_param_rows),
        "negative_samples_per_positive": negative_samples_per_positive,
        "blend_to_learned_weights": blend,
        "old_weights": old_weights,
        "learned_weights": {k: round(v, 6) for k, v in learned_weights.items()},
        "new_weights": {p["ParameterID"]: p["DefaultWeight"] for p in hvti_structure.get("parameters", [])},
        "positive_parameter_averages": {k: round(v, 3) for k, v in pos_avg.items()},
        "negative_parameter_averages": {k: round(v, 3) for k, v in neg_avg.items()},
        "separations": {k: round(v, 3) for k, v in separations.items()},
    }

    log(f"[TUNING] Supervised HVTI weight tuning enabled.")
    log(f"[TUNING] Positive examples used: {len(positive_param_rows):,}")
    log(f"[TUNING] Negative examples used: {len(negative_param_rows):,}")
    log(f"[TUNING] New weights: {tuning_result['new_weights']}")
    return tuning_result


def evaluate_candidate(
    person_a: Dict[str, Any],
    person_b: Dict[str, Any],
    bvh_type: Dict[str, Any],
    hvti_structure: Dict[str, Any],
    calibration: Dict[str, Any],
    run_stamp: str,
) -> Dict[str, Any]:
    fit_ab = estimate_directional_fit(person_a, person_b, bvh_type)
    fit_ba = estimate_directional_fit(person_b, person_a, bvh_type)

    score_ab = score_hvti(fit_ab["parameters"], hvti_structure, calibration, bvh_type["BVHTypeID"])
    score_ba = score_hvti(fit_ba["parameters"], hvti_structure, calibration, bvh_type["BVHTypeID"])

    if score_ba["EstimatedHVTIScore"] > score_ab["EstimatedHVTIScore"]:
        source, target = person_b, person_a
        selected_fit = fit_ba
        hvti_result = score_ba
        direction = "B_TO_A"
    else:
        source, target = person_a, person_b
        selected_fit = fit_ab
        hvti_result = score_ab
        direction = "A_TO_B"

    roi_result = estimate_roi_usd(source, target, bvh_type, hvti_result, hvti_structure, calibration)
    component_scores = estimate_component_scores(
        hvti_result["EstimatedHVTIParameters"],
        selected_fit["evidence"],
        bvh_type["BVHTypeID"],
        roi_result["EstimatedHVITUSDValue"],
    )
    ranking_score = final_ranking_score(hvti_result["EstimatedHVTIScore"], component_scores)

    bvh_id = build_bvh_id(source["PersonID"], target["PersonID"], bvh_type["BVHTypeID"])
    file_suffix = f"{source['PersonID']}_{target['PersonID']}_{bvh_type['BVHTypeID']}_{run_stamp}"
    file_key = stable_id("M", file_suffix, width=12)

    return {
        "source": source,
        "target": target,
        "bvh_type": bvh_type,
        "bvh_id": bvh_id,
        "file_suffix": file_suffix,
        "file_key": file_key,
        "direction": direction,
        "selected_fit": selected_fit,
        "score_ab": score_ab,
        "score_ba": score_ba,
        "fit_ab": fit_ab,
        "fit_ba": fit_ba,
        "hvti_result": hvti_result,
        "roi_result": roi_result,
        "component_scores": component_scores,
        "RankingScore": ranking_score,
    }


def sort_key(candidate: Dict[str, Any]) -> Tuple[float, float, float, float, float, float, float, float, float, float, float]:
    return (
        float(candidate.get("FinalRAINScore", candidate.get("RankingScore", 0))),
        float(candidate.get("PairFinalScore", 0)),
        float(candidate.get("BVHCanonicalScore", 0)),
        float(candidate.get("BVHMarginScore", 0)),
        float(candidate.get("PairPrototypeScore", 0)),
        float(candidate.get("SpecificPairOverlapScore", 0)),
        float(candidate.get("PairSupervisedScore", 0)),
        float(candidate.get("BVHTypeSupervisedScore", 0)),
        float(candidate.get("RankingScore", 0)),
        float(candidate["hvti_result"].get("EstimatedHVTIScore", 0)),
        float(candidate["roi_result"].get("EstimatedExpectedUSDValue", 0)),
    )


def suggestion_pair_key(suggestion: Dict[str, Any]) -> Tuple[str, str]:
    return pair_key(suggestion.get("PersonAName", ""), suggestion.get("PersonBName", ""))


def auto_select_top_k_for_pair_precision(
    suggestions: List[Dict[str, Any]],
    calibration: Dict[str, Any],
    target_pair_precision_pct: float,
    min_suggestions: int,
    max_suggestions: int,
    log,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Select the largest top-K that meets target pair precision. If no K meets the
    target, select the K with best precision, then best recall, then largest K.
    This is backtest-oriented and uses matchesCreated labels when available.
    """
    positive_pairs = calibration.get("positive_pair_names", set())
    total_positive_pairs = len(positive_pairs)

    limit = min(len(suggestions), max_suggestions)
    if limit <= 0 or not positive_pairs:
        return suggestions[:max_suggestions], {
            "enabled": False,
            "reason": "No suggestions or no positive pair labels available.",
            "selected_k": min(len(suggestions), max_suggestions),
        }

    min_k = max(1, min(int(min_suggestions or 1), limit))
    best_meeting_target = None
    best_overall = None

    for k in range(min_k, limit + 1):
        top = suggestions[:k]
        pred_pairs = {suggestion_pair_key(s) for s in top}
        true_pairs = pred_pairs & positive_pairs
        precision = (len(true_pairs) / len(pred_pairs) * 100.0) if pred_pairs else 0.0
        recall = (len(true_pairs) / total_positive_pairs * 100.0) if total_positive_pairs else 0.0
        row = {
            "k": k,
            "unique_pair_count": len(pred_pairs),
            "true_positive_pair_count": len(true_pairs),
            "pair_precision_pct": round(precision, 4),
            "pair_recall_pct": round(recall, 4),
        }

        if precision >= target_pair_precision_pct:
            if best_meeting_target is None or (row["pair_recall_pct"], row["k"]) > (best_meeting_target["pair_recall_pct"], best_meeting_target["k"]):
                best_meeting_target = row

        if best_overall is None or (
            row["pair_precision_pct"],
            row["pair_recall_pct"],
            row["k"],
        ) > (
            best_overall["pair_precision_pct"],
            best_overall["pair_recall_pct"],
            best_overall["k"],
        ):
            best_overall = row

    selected = best_meeting_target or best_overall or {"k": limit}
    selected_k = int(selected["k"])
    selected["enabled"] = True
    selected["target_pair_precision_pct"] = target_pair_precision_pct
    selected["met_target"] = bool(best_meeting_target is not None)
    selected["selection_policy"] = "largest/highest-recall K meeting target; otherwise best precision/recall K"
    log(f"[AUTO-TOP-K] selected_k={selected_k} target={target_pair_precision_pct}% met={selected['met_target']} pair_precision={selected.get('pair_precision_pct')} pair_recall={selected.get('pair_recall_pct')}")
    return suggestions[:selected_k], selected


def build_supervised_ranker_training_rows(
    *,
    profiles: List[Dict[str, Any]],
    bvh_types: List[Dict[str, Any]],
    hvti_structure: Dict[str, Any],
    calibration: Dict[str, Any],
    run_stamp: str,
    max_pairs: int | None,
    work: Dict[str, int],
    progress_every: int,
    progress_seconds: float,
    log,
) -> List[Dict[str, Any]]:
    """
    Build one supervised training row for every candidate pair+BVHType.

    This is intentionally separate from final candidate ranking. It gives the
    V2 ranker full visibility over the candidate universe while allowing the
    final pass to rank by the trained model.
    """
    training_rows: List[Dict[str, Any]] = []
    total_evaluations = work["total_bvh_evaluations"]
    processed = 0
    start_ts = time.time()
    last_progress_ts = start_ts

    log("[RANKER-V2] Building supervised training rows from every candidate pair+BVHType.")

    pair_iter = itertools.combinations(profiles, 2)
    if max_pairs is not None:
        pair_iter = itertools.islice(pair_iter, max_pairs)

    for person_a, person_b in pair_iter:
        for bvh_type in bvh_types:
            if not bvh_type.get("Active", True):
                continue

            processed += 1
            candidate = evaluate_candidate(
                person_a=person_a,
                person_b=person_b,
                bvh_type=bvh_type,
                hvti_structure=hvti_structure,
                calibration=calibration,
                run_stamp=run_stamp,
            )
            training_rows.append(candidate_training_row(candidate, calibration))

            last_progress_ts = maybe_log_progress(
                log=log,
                processed_evaluations=processed,
                total_evaluations=total_evaluations,
                candidate_count=len(training_rows),
                suggestion_count_so_far=0,
                start_ts=start_ts,
                last_progress_ts=last_progress_ts,
                progress_every=progress_every,
                progress_seconds=progress_seconds,
            )

    positive_count = sum(1 for row in training_rows if int(row.get("Label", 0)) == 1)
    log(f"[RANKER-V2] Training rows built: {len(training_rows):,}")
    log(f"[RANKER-V2] Positive labels: {positive_count:,}")
    log(f"[RANKER-V2] Negative labels: {len(training_rows) - positive_count:,}")
    return training_rows


def write_candidate_detail_json(
    candidate: Dict[str, Any],
    output_run_dir: Path,
    values_dir: Path,
    roi_dir: Path,
    hvti_structure: Dict[str, Any],
    calibration: Dict[str, Any],
) -> Tuple[str, str]:
    source = candidate["source"]
    target = candidate["target"]
    bvh_type = candidate["bvh_type"]
    file_suffix = candidate["file_suffix"]
    file_key = candidate["file_key"]
    hvti_result = candidate["hvti_result"]
    roi_result = candidate["roi_result"]
    selected_fit = candidate["selected_fit"]

    hvti_values_path = values_dir / f"V_{file_key}.JSON"
    roi_path = roi_dir / f"R_{file_key}.JSON"

    hvti_values_json = {
        "schema_version": "1.1",
        "object_name": f"HVTIValues_{file_suffix}.JSON",
        "WindowsSafeFileName": f"V_{file_key}.JSON",
        "OriginalRequestedFileName": f"HVTIValues_{file_suffix}.JSON",
        "PersonAID": source["PersonID"],
        "PersonBID": target["PersonID"],
        "PersonAName": source.get("full_name", ""),
        "PersonBName": target.get("full_name", ""),
        "BVHID": candidate["bvh_id"],
        "BVHTypeID": bvh_type["BVHTypeID"],
        "BVHTypeName": bvh_type["BVHTypeName"],
        "Direction": candidate["direction"],
        "EstimatedHVTIParameters": hvti_result["EstimatedHVTIParameters"],
        "EstimatedHVTIScore": hvti_result["EstimatedHVTIScore"],
        "PairMatchScore": candidate["component_scores"]["PairMatchScore"],
        "BVHTypeFitScore": candidate["component_scores"]["BVHTypeFitScore"],
        "ValueScore": candidate["component_scores"]["ValueScore"],
        "RankingScore": candidate["RankingScore"],
        "PairSupervisedScore": candidate.get("PairSupervisedScore", 0.0),
        "BVHTypeSupervisedScore": candidate.get("BVHTypeSupervisedScore", 0.0),
        "FinalRAINScoreRaw": candidate.get("FinalRAINScoreRaw", candidate.get("RankingScore", 0.0)),
        "FinalRAINScore": candidate.get("FinalRAINScore", candidate.get("RankingScore", 0.0)),
        "HardBVHGatePass": candidate.get("HardBVHGatePass", True),
        "HardBVHGateMultiplier": candidate.get("HardBVHGateMultiplier", 1.0),
        "HardBVHGateReason": candidate.get("HardBVHGateReason", ""),
        "EstimatedProbabilityOfTransaction": hvti_result["EstimatedProbabilityOfTransaction"],
        "SignalEvidence": selected_fit["evidence"],
        "BothDirections": {
            "A_TO_B": {
                "PersonAID": candidate["fit_ab"].get("PersonAID", ""),
                "PersonBID": candidate["fit_ab"].get("PersonBID", ""),
                "ScorePreview": candidate["score_ab"],
                "Evidence": candidate["fit_ab"]["evidence"],
            },
            "B_TO_A": {
                "ScorePreview": candidate["score_ba"],
                "Evidence": candidate["fit_ba"]["evidence"],
            },
        },
        "Calibration": hvti_result["Calibration"],
        "KnowledgeUsed": {
            "BVHTypes": "Knowledge/BVHTypes.JSON",
            "HVTIStructure": "Knowledge/HVTIStructure.JSON",
            "RAIN5SignalKG": "Knowledge/RAIN5SignalKG.JSON",
        },
    }

    roi_json = {
        "schema_version": "1.1",
        "object_name": f"HVTIRoI_{file_suffix}.JSON",
        "WindowsSafeFileName": f"R_{file_key}.JSON",
        "OriginalRequestedFileName": f"HVTIRoI_{file_suffix}.JSON",
        "PersonAID": source["PersonID"],
        "PersonBID": target["PersonID"],
        "PersonAName": source.get("full_name", ""),
        "PersonBName": target.get("full_name", ""),
        "BVHID": candidate["bvh_id"],
        "BVHTypeID": bvh_type["BVHTypeID"],
        "BVHTypeName": bvh_type["BVHTypeName"],
        "PairMatchScore": candidate["component_scores"]["PairMatchScore"],
        "BVHTypeFitScore": candidate["component_scores"]["BVHTypeFitScore"],
        "ValueScore": candidate["component_scores"]["ValueScore"],
        "RankingScore": candidate["RankingScore"],
        **roi_result,
    }

    write_json(hvti_values_path, hvti_values_json)
    write_json(roi_path, roi_json)
    return str(hvti_values_path.relative_to(output_run_dir)), str(roi_path.relative_to(output_run_dir))


def candidate_to_suggestion(candidate: Dict[str, Any], hvti_file: str, roi_file: str) -> Dict[str, Any]:
    source = candidate["source"]
    target = candidate["target"]
    bvh_type = candidate["bvh_type"]
    hvti_result = candidate["hvti_result"]
    roi_result = candidate["roi_result"]
    components = candidate["component_scores"]

    return {
        "PersonAName": source.get("full_name", ""),
        "PersonBName": target.get("full_name", ""),
        "PersonAID": source["PersonID"],
        "PersonBID": target["PersonID"],
        "BVHID": candidate["bvh_id"],
        "BVHTypeName": bvh_type["BVHTypeName"],
        "BVHTypeID": bvh_type["BVHTypeID"],
        "EstimatedHVTIParameters": hvti_result["EstimatedHVTIParameters"],
        "EstimatedHVTIScore": hvti_result["EstimatedHVTIScore"],
        "PairMatchScore": components["PairMatchScore"],
        "BVHTypeFitScore": components["BVHTypeFitScore"],
        "ValueScore": components["ValueScore"],
        "RankingScore": candidate["RankingScore"],
        "PairSupervisedProbability": candidate.get("PairSupervisedProbability", 0.0),
        "PairSupervisedScore": candidate.get("PairSupervisedScore", 0.0),
        "BVHTypeSupervisedProbability": candidate.get("BVHTypeSupervisedProbability", 0.0),
        "BVHTypeSupervisedScore": candidate.get("BVHTypeSupervisedScore", 0.0),
        "SupervisedLabelProbability": candidate.get("SupervisedLabelProbability", 0.0),
        "SupervisedLabelScore": candidate.get("SupervisedLabelScore", 0.0),
        "FinalRAINScoreRaw": candidate.get("FinalRAINScoreRaw", candidate.get("RankingScore", 0.0)),
        "FinalRAINScore": candidate.get("FinalRAINScore", candidate.get("RankingScore", 0.0)),
        "HardBVHGatePass": candidate.get("HardBVHGatePass", True),
        "HardBVHGateMultiplier": candidate.get("HardBVHGateMultiplier", 1.0),
        "HardBVHGateScore": candidate.get("HardBVHGateScore", 100.0),
        "HardBVHGateReason": candidate.get("HardBVHGateReason", ""),
        "PairPrototypeScore": candidate.get("PairPrototypeScore", 0.0),
        "PairPrototypeMatch": candidate.get("PairPrototypeMatch", ""),
        "PairPrototypeMatchBVHTypeID": candidate.get("PairPrototypeMatchBVHTypeID", ""),
        "PairPrototypeSimilarityJSON": candidate.get("PairPrototypeSimilarityJSON", {}),
        "SpecificPairOverlapScore": candidate.get("SpecificPairOverlapScore", 0.0),
        "SpecificPairOverlapJSON": candidate.get("SpecificPairOverlapJSON", {}),
        "GenericPatternPenalty": candidate.get("GenericPatternPenalty", 0.0),
        "GenericPatternScore": candidate.get("GenericPatternScore", 0.0),
        "GenericPatternReason": candidate.get("GenericPatternReason", ""),
        "FinalRAINScorePrototypeRaw": candidate.get("FinalRAINScorePrototypeRaw", candidate.get("FinalRAINScoreRaw", candidate.get("RankingScore", 0.0))),
        "FinalRAINScoreBeforeSaturation": candidate.get("FinalRAINScoreBeforeSaturation", candidate.get("FinalRAINScore", candidate.get("RankingScore", 0.0))),
        "PersonSaturationPenalty": candidate.get("PersonSaturationPenalty", 0.0),
        "PairFinalScore": candidate.get("PairFinalScore", 0.0),
        "BVHTypeClassifierProbability": candidate.get("BVHTypeClassifierProbability", 0.0),
        "BVHTypeClassifierScore": candidate.get("BVHTypeClassifierScore", 0.0),
        "BVHSpecificEvidenceScore": candidate.get("BVHSpecificEvidenceScore", 0.0),
        "BVHSpecificEvidenceJSON": candidate.get("BVHSpecificEvidenceJSON", {}),
        "BVHPrototypeTypeScore": candidate.get("BVHPrototypeTypeScore", 0.0),
        "BVHConfusionPenalty": candidate.get("BVHConfusionPenalty", 0.0),
        "BVHConfusionPenaltyReason": candidate.get("BVHConfusionPenaltyReason", ""),
        "BVHCanonicalScoreRaw": candidate.get("BVHCanonicalScoreRaw", 0.0),
        "BVHCanonicalScore": candidate.get("BVHCanonicalScore", 0.0),
        "BestBVHTypeScore": candidate.get("BestBVHTypeScore", 0.0),
        "SecondBestBVHTypeScore": candidate.get("SecondBestBVHTypeScore", 0.0),
        "BVHMarginScore": candidate.get("BVHMarginScore", 0.0),
        "BVHConfidenceBand": candidate.get("BVHConfidenceBand", ""),
        "BVHCanonicalRankWithinPair": candidate.get("BVHCanonicalRankWithinPair", ""),
        "BVHCanonicalKeep": candidate.get("BVHCanonicalKeep", ""),
        "BVHSelectionDecision": candidate.get("BVHSelectionDecision", ""),
        "BVHRecallBoost": candidate.get("BVHRecallBoost", 0.0),
        "BVHRecallBoostApplied": candidate.get("BVHRecallBoostApplied", False),
        "BVHRecallBoostReason": candidate.get("BVHRecallBoostReason", ""),
        "FinalRAINScoreBeforeRecallBoost": candidate.get("FinalRAINScoreBeforeRecallBoost", ""),
        "BVHRecallBalancerAdded": candidate.get("BVHRecallBalancerAdded", False),
        "BVHRecallBalancerReason": candidate.get("BVHRecallBalancerReason", ""),
        "SupervisedFeatureTraceJSON": candidate.get("SupervisedFeatureTraceJSON", {}),
        "EstimatedHVITUSDValue": roi_result["EstimatedHVITUSDValue"],
        "EstimatedHVITUSDValue Explainer": roi_result["EstimatedHVITUSDValue Explainer"],
        "EstimatedExpectedUSDValue": roi_result["EstimatedExpectedUSDValue"],
        "EstimatedProbabilityOfTransaction": hvti_result["EstimatedProbabilityOfTransaction"],
        "HVTIValuesJSONFile": hvti_file,
        "HVTIRoIJSONFile": roi_file,
    }


def run_rain5(
    input_dir: Path,
    output_dir: Path,
    knowledge_dir: Path,
    max_suggestions: int = 250,
    min_hvti_score: float = 50.0,
    min_usd_value: float = 0.0,
    max_pairs: int | None = None,
    profile_sample_pct: float = 100.0,
    progress_every: int = 5000,
    progress_seconds: float = 10.0,
    write_detail_json: bool = True,
    max_bvh_per_pair: int = 1,
    supervised_weight_tuning: bool = True,
    negative_samples_per_positive: int = 5,
    supervised_ranker_v2: bool = True,
    ranker_negative_samples_per_positive: int = 10,
    ranker_epochs: int = 80,
    ranker_learning_rate: float = 0.08,
    ranker_l2: float = 0.001,
    ranker_max_training_rows: int = 5000,
    ranker_random_seed: int = 17,
    ranker_high_value_positive_weight: float = 2.5,
    auto_top_k: bool = False,
    target_pair_precision: float = 50.0,
    auto_top_k_min_suggestions: int = 10,
    prototype_antigeneric: bool = True,
    prototype_max_prototypes: int = 1000,
    person_saturation_penalty: bool = True,
    bvh_precision1: bool = True,
    bvh_selection_mode: str = "canonical_top1",
    bvh_top2_margin_threshold: float = 8.0,
    bvh_top2_min_canonical_score: float = 65.0,
    bvh_recall_balancer: bool = True,
    bvh_recall_floor_spec: str = DEFAULT_FLOOR_SPEC,
    bvh_recall_boost_spec: str = DEFAULT_BOOST_SPEC,
    bvh_recall_boost_min_canonical: float = 50.0,
    bvh_recall_floor_min_score: float = 0.0,
    bvh_recall_floor_min_canonical: float = 0.0,
    bvh_recall_allow_precision_drop_pct: float = 0.0,
    bvh_recall_use_backtest_precision_guard: bool = True,
    write_bvh_recall_pool: bool = False,
) -> Dict[str, Any]:
    run_stamp = now_stamp()
    output_run_dir = output_dir / f"OUTPUT_{run_stamp}"
    output_run_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_run_dir / "RAIN5_RunLog.txt"
    log = make_logger(log_path)

    start_ts = time.time()
    log("[START] RAIN5.0 match suggestion run started")
    log(f"[CONFIG] input_dir={input_dir}")
    log(f"[CONFIG] output_dir={output_dir}")
    log(f"[CONFIG] knowledge_dir={knowledge_dir}")
    log(f"[CONFIG] max_suggestions={max_suggestions}")
    log(f"[CONFIG] min_hvti_score={min_hvti_score}")
    log(f"[CONFIG] min_usd_value={min_usd_value}")
    log(f"[CONFIG] max_pairs={max_pairs}")
    log(f"[CONFIG] profile_sample_pct={profile_sample_pct}")
    log(f"[CONFIG] progress_every={progress_every}")
    log(f"[CONFIG] progress_seconds={progress_seconds}")
    log(f"[CONFIG] write_detail_json={write_detail_json}")
    log(f"[CONFIG] max_bvh_per_pair={max_bvh_per_pair}")
    log(f"[CONFIG] supervised_weight_tuning={supervised_weight_tuning}")
    log(f"[CONFIG] negative_samples_per_positive={negative_samples_per_positive}")
    log(f"[CONFIG] supervised_ranker_v2={supervised_ranker_v2}")
    log(f"[CONFIG] ranker_negative_samples_per_positive={ranker_negative_samples_per_positive}")
    log(f"[CONFIG] ranker_epochs={ranker_epochs}")
    log(f"[CONFIG] ranker_learning_rate={ranker_learning_rate}")
    log(f"[CONFIG] ranker_l2={ranker_l2}")
    log(f"[CONFIG] ranker_max_training_rows={ranker_max_training_rows}")
    log(f"[CONFIG] ranker_random_seed={ranker_random_seed}")
    log(f"[CONFIG] ranker_high_value_positive_weight={ranker_high_value_positive_weight}")
    log(f"[CONFIG] auto_top_k={auto_top_k}")
    log(f"[CONFIG] target_pair_precision={target_pair_precision}")
    log(f"[CONFIG] auto_top_k_min_suggestions={auto_top_k_min_suggestions}")
    log(f"[CONFIG] prototype_antigeneric={prototype_antigeneric}")
    log(f"[CONFIG] prototype_max_prototypes={prototype_max_prototypes}")
    log(f"[CONFIG] person_saturation_penalty={person_saturation_penalty}")
    log(f"[CONFIG] bvh_precision1={bvh_precision1}")
    log(f"[CONFIG] bvh_selection_mode={bvh_selection_mode}")
    log(f"[CONFIG] bvh_top2_margin_threshold={bvh_top2_margin_threshold}")
    log(f"[CONFIG] bvh_top2_min_canonical_score={bvh_top2_min_canonical_score}")
    log(f"[CONFIG] bvh_recall_balancer={bvh_recall_balancer}")
    log(f"[CONFIG] bvh_recall_floor_spec={bvh_recall_floor_spec}")
    log(f"[CONFIG] bvh_recall_boost_spec={bvh_recall_boost_spec}")
    log(f"[CONFIG] bvh_recall_boost_min_canonical={bvh_recall_boost_min_canonical}")
    log(f"[CONFIG] bvh_recall_floor_min_score={bvh_recall_floor_min_score}")
    log(f"[CONFIG] bvh_recall_floor_min_canonical={bvh_recall_floor_min_canonical}")
    log(f"[CONFIG] bvh_recall_allow_precision_drop_pct={bvh_recall_allow_precision_drop_pct}")
    log(f"[CONFIG] bvh_recall_use_backtest_precision_guard={bvh_recall_use_backtest_precision_guard}")
    log(f"[CONFIG] write_bvh_recall_pool={write_bvh_recall_pool}")

    copied_inputs = copy_inputs_to_output(input_dir, output_run_dir)

    cil_profiles_path = find_latest("CILProfiles*.csv", input_dir)
    if not cil_profiles_path:
        raise FileNotFoundError(f"No CILProfiles*.csv found in {input_dir}")

    matches_created_path = find_latest("matchesCreated*.csv", input_dir)

    bvh_types = read_json(knowledge_dir / "BVHTypes.JSON")["bvh_types"]
    hvti_structure = read_json(knowledge_dir / "HVTIStructure.JSON")
    normalise_weights(hvti_structure)

    profiles_all = load_profiles(cil_profiles_path)
    profiles = apply_profile_sample(profiles_all, profile_sample_pct, log)

    calibration = load_labelled_matches(matches_created_path)

    bvh_recall_floors = parse_bvh_floor_spec(bvh_recall_floor_spec)
    bvh_recall_boosts = parse_bvh_boost_spec(bvh_recall_boost_spec)
    log(f"[BVH-RECALL] Floors: {bvh_recall_floors}")
    log(f"[BVH-RECALL] Boosts: {bvh_recall_boosts}")

    prototype_model = {"enabled": False, "reason": "Prototype anti-generic scoring disabled."}
    if prototype_antigeneric:
        prototype_model = build_pair_prototype_model(
            profiles=profiles,
            calibration=calibration,
            max_prototypes=prototype_max_prototypes,
        )
        write_json(output_run_dir / "PairPrototypeModel.JSON", serialise_pair_prototype_model(prototype_model))
        log(f"[PROTOTYPE] Pair prototype model enabled: {prototype_model.get('enabled')}")
        log(f"[PROTOTYPE] Prototype count: {prototype_model.get('prototype_count', 0)}")
    else:
        write_json(output_run_dir / "PairPrototypeModel.JSON", serialise_pair_prototype_model(prototype_model))

    tuning_result = {"enabled": False, "reason": "Supervised tuning disabled."}
    if supervised_weight_tuning:
        tuning_result = tune_hvti_weights_from_labels(
            profiles=profiles,
            bvh_types=bvh_types,
            hvti_structure=hvti_structure,
            calibration=calibration,
            negative_samples_per_positive=negative_samples_per_positive,
            log=log,
        )

    write_json(output_run_dir / "HVTIStructure_Tuned.JSON", hvti_structure)
    write_json(output_run_dir / "SupervisedWeightTuning.JSON", tuning_result)

    values_dir = output_run_dir / "HVTIValues"
    roi_dir = output_run_dir / "HVTIRoI"
    values_dir.mkdir(parents=True, exist_ok=True)
    roi_dir.mkdir(parents=True, exist_ok=True)

    work = estimate_total_work(len(profiles), len(bvh_types), max_pairs)
    total_evaluations = work["total_bvh_evaluations"]

    log(f"[INPUT] CILProfiles={cil_profiles_path}")
    log(f"[INPUT] matchesCreated={matches_created_path if matches_created_path else 'None'}")
    log(f"[INPUT] original_profile_count={len(profiles_all)}")
    log(f"[INPUT] active_profile_count={len(profiles)}")
    log(f"[INPUT] bvh_type_count={len(bvh_types)}")
    log(f"[PLAN] total_pairs_all={work['total_pairs_all']:,}")
    log(f"[PLAN] total_pairs_to_process={work['total_pairs']:,}")
    log(f"[PLAN] total_bvh_evaluations={total_evaluations:,}")

    supervised_ranker_model: Dict[str, Any] = {
        "enabled": False,
        "reason": "Supervised Ranker V2 disabled.",
    }
    supervised_training_row_count = 0
    supervised_positive_count = 0
    supervised_pair_positive_count = 0
    training_rows: List[Dict[str, Any]] = []

    if supervised_ranker_v2:
        training_rows = build_supervised_ranker_training_rows(
            profiles=profiles,
            bvh_types=bvh_types,
            hvti_structure=hvti_structure,
            calibration=calibration,
            run_stamp=run_stamp,
            max_pairs=max_pairs,
            work=work,
            progress_every=progress_every,
            progress_seconds=progress_seconds,
            log=log,
        )
        supervised_training_row_count = len(training_rows)
        supervised_positive_count = sum(1 for row in training_rows if int(row.get("BVHTypeLabel", row.get("Label", 0))) == 1)
        supervised_pair_positive_count = sum(1 for row in training_rows if int(row.get("PairLabel", 0)) == 1)

        supervised_ranker_model = train_precision50_models(
            training_rows,
            negative_samples_per_positive=ranker_negative_samples_per_positive,
            epochs=ranker_epochs,
            learning_rate=ranker_learning_rate,
            l2=ranker_l2,
            max_training_rows=ranker_max_training_rows,
            random_seed=ranker_random_seed,
            high_value_positive_weight=ranker_high_value_positive_weight,
        )
        write_json(output_run_dir / "SupervisedRankerV2.JSON", supervised_ranker_model)
        log(f"[PRECISION50] Model enabled: {supervised_ranker_model.get('enabled')}")
        log(f"[PRECISION50] Pair model diagnostics: {supervised_ranker_model.get('pair_model', {}).get('diagnostics', {})}")
        log(f"[PRECISION50] BVH model diagnostics: {supervised_ranker_model.get('bvh_type_model', {}).get('diagnostics', {})}")
    else:
        write_json(output_run_dir / "SupervisedRankerV2.JSON", supervised_ranker_model)

    bvh_precision_model: Dict[str, Any] = {"enabled": False, "reason": "BVH Precision@1 disabled."}
    if bvh_precision1 and supervised_ranker_v2 and training_rows:
        bvh_precision_model = train_bvh_precision_models(
            training_rows=training_rows,
            bvh_type_ids=[b.get("BVHTypeID", "") for b in bvh_types],
            negative_samples_per_positive=ranker_negative_samples_per_positive,
            epochs=ranker_epochs,
            learning_rate=ranker_learning_rate,
            l2=ranker_l2,
            max_training_rows=ranker_max_training_rows,
            random_seed=ranker_random_seed,
            high_value_positive_weight=ranker_high_value_positive_weight,
        )
        write_json(output_run_dir / "BVHPrecision1Model.JSON", bvh_precision_model)
        log(f"[BVH-PRECISION1] Model enabled: {bvh_precision_model.get('enabled')}")
        log(f"[BVH-PRECISION1] Enabled classifiers: {bvh_precision_model.get('enabled_classifier_count', 0)} / {bvh_precision_model.get('bvh_type_count', 0)}")
    else:
        write_json(output_run_dir / "BVHPrecision1Model.JSON", bvh_precision_model)

    suggestions: List[Dict[str, Any]] = []
    all_rank_rows: List[Dict[str, Any]] = []
    candidate_count = 0
    filtered_candidate_count = 0
    processed_evaluations = 0
    detail_file_count = 0
    last_progress_ts = start_ts

    pair_iter = itertools.combinations(profiles, 2)
    if max_pairs is not None:
        pair_iter = itertools.islice(pair_iter, max_pairs)

    for person_a, person_b in pair_iter:
        pair_candidates: List[Dict[str, Any]] = []

        for bvh_type in bvh_types:
            if not bvh_type.get("Active", True):
                continue

            processed_evaluations += 1
            candidate = evaluate_candidate(
                person_a=person_a,
                person_b=person_b,
                bvh_type=bvh_type,
                hvti_structure=hvti_structure,
                calibration=calibration,
                run_stamp=run_stamp,
            )

            if supervised_ranker_v2 and supervised_ranker_model.get("enabled"):
                candidate.update(apply_supervised_scores(candidate, supervised_ranker_model))
            else:
                candidate["PairSupervisedProbability"] = 0.0
                candidate["PairSupervisedScore"] = 0.0
                candidate["BVHTypeSupervisedProbability"] = 0.0
                candidate["BVHTypeSupervisedScore"] = 0.0
                candidate["SupervisedLabelProbability"] = 0.0
                candidate["SupervisedLabelScore"] = 0.0
                candidate["FinalRAINScoreRaw"] = candidate.get("RankingScore", 0.0)
                candidate["FinalRAINScore"] = candidate.get("RankingScore", 0.0)
                candidate["HardBVHGatePass"] = True
                candidate["HardBVHGateMultiplier"] = 1.0
                candidate["HardBVHGateScore"] = 100.0
                candidate["HardBVHGateReason"] = "Supervised ranker disabled."
                candidate["SupervisedFeatureTraceJSON"] = {}

            if prototype_antigeneric and prototype_model.get("enabled"):
                candidate.update(apply_prototype_antigeneric_scores(candidate, prototype_model))
            else:
                candidate["PairPrototypeScore"] = 0.0
                candidate["PairPrototypeMatch"] = ""
                candidate["PairPrototypeMatchBVHTypeID"] = ""
                candidate["PairPrototypeSimilarityJSON"] = {}
                candidate["SpecificPairOverlapScore"] = 0.0
                candidate["SpecificPairOverlapJSON"] = {}
                candidate["GenericPatternPenalty"] = 0.0
                candidate["GenericPatternScore"] = 0.0
                candidate["GenericPatternReason"] = "Prototype anti-generic scoring disabled."
                candidate["FinalRAINScoreBeforeSaturation"] = candidate.get("FinalRAINScore", candidate.get("RankingScore", 0.0))
                candidate["PersonSaturationPenalty"] = 0.0

            if bvh_precision1 and bvh_precision_model.get("enabled"):
                candidate.update(apply_bvh_precision_scores(candidate, bvh_precision_model))
            else:
                candidate["PairFinalScore"] = candidate.get("FinalRAINScoreBeforeSaturation", candidate.get("FinalRAINScore", 0.0))
                candidate["BVHTypeClassifierProbability"] = 0.0
                candidate["BVHTypeClassifierScore"] = 0.0
                candidate["BVHSpecificEvidenceScore"] = 0.0
                candidate["BVHSpecificEvidenceJSON"] = {}
                candidate["BVHPrototypeTypeScore"] = 0.0
                candidate["BVHConfusionPenalty"] = 0.0
                candidate["BVHConfusionPenaltyReason"] = "BVH Precision@1 disabled."
                candidate["BVHCanonicalScoreRaw"] = candidate.get("BVHTypeSupervisedScore", 0.0)
                candidate["BVHCanonicalScore"] = candidate.get("BVHTypeSupervisedScore", 0.0)

            if bvh_recall_balancer:
                candidate.update(apply_bvh_recall_boost(
                    candidate,
                    boost_by_type=bvh_recall_boosts,
                    min_canonical_score=bvh_recall_boost_min_canonical,
                ))
            else:
                candidate["BVHRecallBoost"] = 0.0
                candidate["BVHRecallBoostApplied"] = False
                candidate["BVHRecallBoostReason"] = "BVH recall balancer disabled."

            keep_candidate = (
                candidate["hvti_result"]["EstimatedHVTIScore"] >= min_hvti_score
                and candidate["roi_result"]["EstimatedHVITUSDValue"] >= min_usd_value
            )

            all_rank_rows.append(candidate_rank_row(candidate, keep_candidate))

            if keep_candidate:
                candidate_count += 1
                pair_candidates.append(candidate)
            else:
                filtered_candidate_count += 1

            last_progress_ts = maybe_log_progress(
                log=log,
                processed_evaluations=processed_evaluations,
                total_evaluations=total_evaluations,
                candidate_count=candidate_count,
                suggestion_count_so_far=len(suggestions),
                start_ts=start_ts,
                last_progress_ts=last_progress_ts,
                progress_every=progress_every,
                progress_seconds=progress_seconds,
            )

        pair_candidates, bvh_pair_selection_summary = select_bvh_candidates_for_pair(
            pair_candidates,
            mode=bvh_selection_mode if bvh_precision1 else "legacy",
            top2_margin_threshold=bvh_top2_margin_threshold,
            top2_min_canonical_score=bvh_top2_min_canonical_score,
            legacy_max_bvh_per_pair=max_bvh_per_pair,
        )
        pair_candidates.sort(key=sort_key, reverse=True)

        for candidate in pair_candidates:
            # Defer detail JSON writes until after final ranking.
            #
            # Writing thousands of small HVTIValues/HVTIRoI files during the scoring
            # loop is slow and can cause Windows / Google Drive file-stream errors.
            # Keep the candidate object temporarily, rank first, and only write detail
            # JSON for the final top suggestions.
            suggestion = candidate_to_suggestion(candidate, "", "")
            suggestion["_candidate"] = candidate
            suggestions.append(suggestion)

        if len(suggestions) > max_suggestions * 4:
            suggestions.sort(
                key=lambda x: (
                    float(x.get("FinalRAINScore", x.get("RankingScore", 0))),
                    float(x.get("PairFinalScore", 0)),
                    float(x.get("BVHCanonicalScore", 0)),
                    float(x.get("BVHMarginScore", 0)),
                    float(x.get("PairPrototypeScore", 0)),
                    float(x.get("SpecificPairOverlapScore", 0)),
                    float(x.get("PairSupervisedScore", 0)),
                    float(x.get("BVHTypeSupervisedScore", 0)),
                    float(x.get("RankingScore", 0)),
                    float(x.get("EstimatedHVTIScore", 0)),
                    float(x.get("EstimatedExpectedUSDValue", 0)),
                    float(x.get("EstimatedHVITUSDValue", 0)),
                ),
                reverse=True,
            )
            suggestions = suggestions[: max_suggestions * 2]

    saturation_summary: Dict[str, Any] = {"enabled": False, "reason": "Person saturation penalty disabled."}
    if prototype_antigeneric and person_saturation_penalty:
        saturation_summary = apply_person_saturation_penalty_to_suggestions(suggestions)

    suggestions.sort(
        key=lambda x: (
            float(x.get("FinalRAINScore", x.get("RankingScore", 0))),
            float(x.get("PairFinalScore", 0)),
            float(x.get("BVHCanonicalScore", 0)),
            float(x.get("BVHMarginScore", 0)),
            float(x.get("PairPrototypeScore", 0)),
            float(x.get("SpecificPairOverlapScore", 0)),
            float(x.get("PairSupervisedScore", 0)),
            float(x.get("BVHTypeSupervisedScore", 0)),
            float(x.get("RankingScore", 0)),
            float(x.get("EstimatedHVTIScore", 0)),
            float(x.get("EstimatedExpectedUSDValue", 0)),
            float(x.get("EstimatedHVITUSDValue", 0)),
        ),
        reverse=True,
    )
    auto_top_k_summary: Dict[str, Any] = {
        "enabled": False,
        "selected_k": min(len(suggestions), max_suggestions),
        "reason": "auto_top_k disabled",
    }
    bvh_recall_ranked_pool = list(suggestions)

    if write_bvh_recall_pool:
        pool_export = []
        for s in bvh_recall_ranked_pool[: min(len(bvh_recall_ranked_pool), 5000)]:
            ss = dict(s)
            ss.pop("_candidate", None)
            pool_export.append(ss)
        write_json(output_run_dir / "BVHRecallBalancerPool.JSON", {
            "schema_version": "1.0",
            "description": "Canonical ranked candidate pool before auto-top-K and BVH floor balancing.",
            "candidate_count": len(bvh_recall_ranked_pool),
            "rows": pool_export,
        })

    if auto_top_k:
        suggestions, auto_top_k_summary = auto_select_top_k_for_pair_precision(
            suggestions=suggestions,
            calibration=calibration,
            target_pair_precision_pct=target_pair_precision,
            min_suggestions=auto_top_k_min_suggestions,
            max_suggestions=max_suggestions,
            log=log,
        )
    else:
        suggestions = suggestions[:max_suggestions]

    bvh_recall_balancer_summary: Dict[str, Any] = {"enabled": False, "reason": "BVH recall balancer disabled."}
    if bvh_recall_balancer:
        suggestions, bvh_recall_balancer_summary = balance_bvh_type_floors(
            ranked_pool=bvh_recall_ranked_pool,
            selected=suggestions,
            calibration=calibration,
            floors_by_type=bvh_recall_floors,
            target_pair_precision_pct=target_pair_precision,
            max_suggestions=max_suggestions,
            min_candidate_score=bvh_recall_floor_min_score,
            min_bvh_canonical_score=bvh_recall_floor_min_canonical,
            allow_precision_drop_pct=bvh_recall_allow_precision_drop_pct,
            use_backtest_precision_guard=bvh_recall_use_backtest_precision_guard,
        )
        log(f"[BVH-RECALL] Balancer summary: {bvh_recall_balancer_summary}")

    if write_detail_json:
        log(f"[DETAIL] Writing detail JSON files only for final top {len(suggestions):,} suggestions.")
        detail_file_count = 0
        for suggestion in suggestions:
            candidate = suggestion.get("_candidate")
            if not candidate:
                continue
            hvti_file, roi_file = write_candidate_detail_json(
                candidate=candidate,
                output_run_dir=output_run_dir,
                values_dir=values_dir,
                roi_dir=roi_dir,
                hvti_structure=hvti_structure,
                calibration=calibration,
            )
            suggestion["HVTIValuesJSONFile"] = hvti_file
            suggestion["HVTIRoIJSONFile"] = roi_file
            detail_file_count += 2
        log(f"[DETAIL] Detail JSON complete: {detail_file_count:,} files written.")
    else:
        detail_file_count = 0

    for suggestion in suggestions:
        suggestion.pop("_candidate", None)

    label_coverage_rows = build_label_coverage_report(
        calibration=calibration,
        rank_rows=all_rank_rows,
        final_suggestions=suggestions,
        max_suggestions=max_suggestions,
    )
    label_coverage_paths = write_label_coverage_report(output_run_dir, label_coverage_rows)
    log(f"[LABEL-COVERAGE] rows={len(label_coverage_rows):,} csv={label_coverage_paths.get('csv')}")

    match_suggestions = {
        "schema_version": "1.1",
        "object_name": "matchSuggestions.JSON",
        "run_stamp": run_stamp,
        "input_files": {
            "CILProfiles": str(cil_profiles_path),
            "matchesCreated": str(matches_created_path) if matches_created_path else None,
            "copied_inputs": copied_inputs,
        },
        "algorithm": {
            "name": "RAIN5.0",
            "description": (
                "Explores every profile pair and every BVHType, estimates HVTI parameters "
                "from profile data, applies supervised weight tuning from matchesCreated, "
                "scores pair/BVH/value components, limits BVHTypes per pair, estimates "
                "potential USD value and ranks introduction opportunities."
            ),
            "ranking_formula": "FinalRAINScore = 0.70 * PairFinalScore + 0.30 * BVHCanonicalScore - PersonSaturationPenalty",
            "pair_formula": "PairFinalScore = 0.45 * PairPrototypeScore + 0.30 * SpecificPairOverlapScore + 0.25 * PairSupervisedScore - GenericPatternPenalty",
            "bvh_canonical_formula": "BVHCanonicalScore = 0.35 * BVHTypeClassifierScore + 0.25 * BVHSpecificEvidenceScore + 0.20 * BVHPrototypeTypeScore + 0.10 * BVHTypeSupervisedScore + 0.10 * HardBVHGateScore - BVHConfusionPenalty",
            "prototype_antigeneric_formula": "0.35 * PairPrototypeScore + 0.25 * SpecificPairOverlapScore + 0.20 * PairSupervisedScore + 0.10 * BVHTypeSupervisedScore + 0.10 * ValueScore",
            "precision50_formula": "0.50 * PairSupervisedScore + 0.30 * BVHTypeSupervisedScore + 0.10 * ValueScore + 0.10 * RankingScore",
            "legacy_ranking_formula": "0.45 * EstimatedHVTIScore + 0.35 * ValueScore + 0.20 * BVHTypeFitScore",
            "min_hvti_score": min_hvti_score,
            "min_usd_value": min_usd_value,
            "max_suggestions": max_suggestions,
            "max_pairs": max_pairs,
            "profile_sample_pct": profile_sample_pct,
            "progress_every": progress_every,
            "progress_seconds": progress_seconds,
            "write_detail_json": write_detail_json,
            "detail_json_write_policy": "deferred_until_after_final_ranking",
            "max_bvh_per_pair": max_bvh_per_pair,
            "supervised_weight_tuning": supervised_weight_tuning,
            "negative_samples_per_positive": negative_samples_per_positive,
            "supervised_ranker_v2": supervised_ranker_v2,
            "ranker_negative_samples_per_positive": ranker_negative_samples_per_positive,
            "ranker_epochs": ranker_epochs,
            "ranker_learning_rate": ranker_learning_rate,
            "ranker_l2": ranker_l2,
            "ranker_max_training_rows": ranker_max_training_rows,
            "ranker_random_seed": ranker_random_seed,
            "ranker_high_value_positive_weight": ranker_high_value_positive_weight,
            "auto_top_k": auto_top_k,
            "target_pair_precision": target_pair_precision,
            "auto_top_k_min_suggestions": auto_top_k_min_suggestions,
            "prototype_antigeneric": prototype_antigeneric,
            "prototype_max_prototypes": prototype_max_prototypes,
            "person_saturation_penalty": person_saturation_penalty,
            "bvh_precision1": bvh_precision1,
            "bvh_selection_mode": bvh_selection_mode,
            "bvh_top2_margin_threshold": bvh_top2_margin_threshold,
            "bvh_top2_min_canonical_score": bvh_top2_min_canonical_score,
            "bvh_recall_balancer": bvh_recall_balancer,
            "bvh_recall_floor_spec": bvh_recall_floor_spec,
            "bvh_recall_boost_spec": bvh_recall_boost_spec,
            "bvh_recall_boost_min_canonical": bvh_recall_boost_min_canonical,
            "bvh_recall_floor_min_score": bvh_recall_floor_min_score,
            "bvh_recall_floor_min_canonical": bvh_recall_floor_min_canonical,
            "bvh_recall_allow_precision_drop_pct": bvh_recall_allow_precision_drop_pct,
            "bvh_recall_use_backtest_precision_guard": bvh_recall_use_backtest_precision_guard,
            "write_bvh_recall_pool": write_bvh_recall_pool,
        },
        "supervised_weight_tuning": tuning_result,
        "supervised_ranker_v2_model": supervised_ranker_model,
        "pair_prototype_model_summary": {
            "enabled": prototype_model.get("enabled", False),
            "method": prototype_model.get("method", ""),
            "prototype_count": prototype_model.get("prototype_count", 0),
            "model_file": str(output_run_dir / "PairPrototypeModel.JSON"),
        },
        "bvh_precision1_model_summary": bvh_precision_model_summary(bvh_precision_model),
        "auto_top_k_summary": auto_top_k_summary,
        "person_saturation_summary": saturation_summary,
        "label_coverage_report": label_coverage_paths,
        "summary": {
            "original_profile_count": len(profiles_all),
            "active_profile_count": len(profiles),
            "bvh_type_count": len(bvh_types),
            "total_pairs_all": work["total_pairs_all"],
            "total_pairs_processed": work["total_pairs"],
            "total_bvh_evaluations_planned": total_evaluations,
            "total_bvh_evaluations_processed": processed_evaluations,
            "candidate_count_after_filters_before_pair_cap": candidate_count,
            "filtered_candidate_count": filtered_candidate_count,
            "suggestion_count": len(suggestions),
            "detail_file_count": detail_file_count,
            "labelled_match_count": calibration.get("label_count", 0),
            "supervised_ranker_training_row_count": supervised_training_row_count,
            "supervised_ranker_pair_positive_count": supervised_pair_positive_count,
            "supervised_ranker_bvh_positive_count": supervised_positive_count,
            "supervised_ranker_positive_count": supervised_positive_count,
            "supervised_ranker_enabled": supervised_ranker_model.get("enabled", False),
            "auto_top_k_summary": auto_top_k_summary,
            "prototype_antigeneric_enabled": prototype_model.get("enabled", False),
            "pair_prototype_count": prototype_model.get("prototype_count", 0),
            "person_saturation_summary": saturation_summary,
            "label_coverage_report": label_coverage_paths,
            "label_coverage_row_count": len(label_coverage_rows),
            "bvh_precision1_enabled": bvh_precision_model.get("enabled", False),
            "bvh_precision1_enabled_classifier_count": bvh_precision_model.get("enabled_classifier_count", 0),
            "bvh_selection_mode": bvh_selection_mode,
            "bvh_recall_balancer_enabled": bvh_recall_balancer,
            "bvh_recall_floors": bvh_recall_floors,
            "bvh_recall_boosts": bvh_recall_boosts,
            "bvh_recall_balancer_summary": bvh_recall_balancer_summary,
            "bvh_type_counts_after_selection": count_by_bvh(suggestions),
            "bvh_recall_pool_file": str(output_run_dir / "BVHRecallBalancerPool.JSON") if write_bvh_recall_pool else "",
            "elapsed_seconds": round(time.time() - start_ts, 2),
        },
        "matchSuggestions": suggestions,
    }

    match_suggestions_path = output_run_dir / "matchSuggestions.JSON"
    write_json(match_suggestions_path, match_suggestions)

    elapsed = time.time() - start_ts
    log(f"[DONE] RAIN5.0 run complete in {format_duration(elapsed)}")
    log(f"[OUT] matchSuggestions.JSON={match_suggestions_path}")
    log(f"[OUT] HVTIValues folder={values_dir}")
    log(f"[OUT] HVTIRoI folder={roi_dir}")
    log(f"[OUT] RAIN5_RunLog.txt={log_path}")
    log(f"[OUT] HVTIStructure_Tuned.JSON={output_run_dir / 'HVTIStructure_Tuned.JSON'}")
    log(f"[OUT] SupervisedWeightTuning.JSON={output_run_dir / 'SupervisedWeightTuning.JSON'}")
    log(f"[OUT] SupervisedRankerV2.JSON={output_run_dir / 'SupervisedRankerV2.JSON'}")
    log(f"[OUT] PairPrototypeModel.JSON={output_run_dir / 'PairPrototypeModel.JSON'}")
    log(f"[OUT] BVHPrecision1Model.JSON={output_run_dir / 'BVHPrecision1Model.JSON'}")
    log(f"[OUT] LabelCoverageReport.csv={label_coverage_paths.get('csv')}")
    log(f"[OUT] LabelCoverageReport.JSON={label_coverage_paths.get('json')}")
    if write_bvh_recall_pool:
        log(f"[OUT] BVHRecallBalancerPool.JSON={output_run_dir / 'BVHRecallBalancerPool.JSON'}")
    log(f"[METRIC] supervised_ranker_training_rows={supervised_training_row_count:,}")
    log(f"[METRIC] supervised_ranker_pair_positive_rows={supervised_pair_positive_count:,}")
    log(f"[METRIC] supervised_ranker_bvh_positive_rows={supervised_positive_count:,}")
    log(f"[METRIC] auto_top_k_summary={auto_top_k_summary}")
    log(f"[METRIC] prototype_antigeneric_enabled={prototype_model.get('enabled', False)}")
    log(f"[METRIC] pair_prototype_count={prototype_model.get('prototype_count', 0):,}")
    log(f"[METRIC] person_saturation_summary={saturation_summary}")
    log(f"[METRIC] label_coverage_rows={len(label_coverage_rows):,}")
    log(f"[METRIC] bvh_precision1_enabled={bvh_precision_model.get('enabled', False)}")
    log(f"[METRIC] bvh_precision1_enabled_classifier_count={bvh_precision_model.get('enabled_classifier_count', 0):,}")
    log(f"[METRIC] bvh_selection_mode={bvh_selection_mode}")
    log(f"[METRIC] bvh_recall_balancer_enabled={bvh_recall_balancer}")
    log(f"[METRIC] bvh_recall_balancer_summary={bvh_recall_balancer_summary}")
    log(f"[METRIC] bvh_type_counts_after_selection={count_by_bvh(suggestions)}")
    log(f"[METRIC] candidates_after_filters_before_pair_cap={candidate_count:,}")
    log(f"[METRIC] suggestions={len(suggestions):,}")
    log(f"[METRIC] detail_files_written={detail_file_count:,}")

    manifest = {
        "run_stamp": run_stamp,
        "output_run_dir": str(output_run_dir),
        "matchSuggestions": str(match_suggestions_path),
        "HVTIValues_folder": str(values_dir),
        "HVTIRoI_folder": str(roi_dir),
        "RAIN5_RunLog": str(log_path),
        "HVTIStructure_Tuned": str(output_run_dir / "HVTIStructure_Tuned.JSON"),
        "SupervisedWeightTuning": str(output_run_dir / "SupervisedWeightTuning.JSON"),
        "SupervisedRankerV2": str(output_run_dir / "SupervisedRankerV2.JSON"),
        "PairPrototypeModel": str(output_run_dir / "PairPrototypeModel.JSON"),
        "BVHPrecision1Model": str(output_run_dir / "BVHPrecision1Model.JSON"),
        "BVHRecallBalancerPool": str(output_run_dir / "BVHRecallBalancerPool.JSON") if write_bvh_recall_pool else "",
        "LabelCoverageReportCSV": label_coverage_paths.get("csv"),
        "LabelCoverageReportJSON": label_coverage_paths.get("json"),
        "original_profile_count": len(profiles_all),
        "active_profile_count": len(profiles),
        "profile_sample_pct": profile_sample_pct,
        "bvh_type_count": len(bvh_types),
        "suggestion_count": len(suggestions),
        "candidate_count_after_filters_before_pair_cap": candidate_count,
        "filtered_candidate_count": filtered_candidate_count,
        "detail_file_count": detail_file_count,
        "max_bvh_per_pair": max_bvh_per_pair,
        "supervised_weight_tuning": supervised_weight_tuning,
        "supervised_ranker_v2": supervised_ranker_v2,
        "supervised_ranker_training_row_count": supervised_training_row_count,
        "supervised_ranker_pair_positive_count": supervised_pair_positive_count,
        "supervised_ranker_bvh_positive_count": supervised_positive_count,
        "supervised_ranker_positive_count": supervised_positive_count,
        "auto_top_k": auto_top_k,
        "target_pair_precision": target_pair_precision,
        "auto_top_k_summary": auto_top_k_summary,
        "prototype_antigeneric": prototype_antigeneric,
        "prototype_model_enabled": prototype_model.get("enabled", False),
        "pair_prototype_count": prototype_model.get("prototype_count", 0),
        "person_saturation_penalty": person_saturation_penalty,
        "person_saturation_summary": saturation_summary,
        "bvh_precision1": bvh_precision1,
        "bvh_precision1_model_enabled": bvh_precision_model.get("enabled", False),
        "bvh_precision1_enabled_classifier_count": bvh_precision_model.get("enabled_classifier_count", 0),
        "bvh_selection_mode": bvh_selection_mode,
        "bvh_top2_margin_threshold": bvh_top2_margin_threshold,
        "bvh_top2_min_canonical_score": bvh_top2_min_canonical_score,
        "bvh_recall_balancer": bvh_recall_balancer,
        "bvh_recall_floors": bvh_recall_floors,
        "bvh_recall_boosts": bvh_recall_boosts,
        "bvh_recall_balancer_summary": bvh_recall_balancer_summary,
        "bvh_type_counts_after_selection": count_by_bvh(suggestions),
        "write_bvh_recall_pool": write_bvh_recall_pool,
        "label_coverage_row_count": len(label_coverage_rows),
        "elapsed_seconds": round(elapsed, 2),
        "copied_inputs": copied_inputs,
    }
    write_json(output_run_dir / "RAIN5_RunManifest.JSON", manifest)
    return manifest
