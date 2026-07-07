from __future__ import annotations
from typing import Any, Dict, List, Tuple
from .utils import safe_float, normalise_text
from .calibration import pair_key


DEFAULT_FLOOR_SPEC = "PARTNERSHIP:5,TRUSTPATH:5,RECRUITMENT:5,SUPPLIERMATCH:5"
DEFAULT_BOOST_SPEC = "PARTNERSHIP:4,TRUSTPATH:6,RECRUITMENT:4,SUPPLIERMATCH:4"


def parse_bvh_number_spec(value: str | None, default: str = "") -> Dict[str, float]:
    text = (value or default or "").strip()
    out: Dict[str, float] = {}
    if not text:
        return out
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            out[part.strip().upper()] = 1.0
            continue
        k, v = part.split(":", 1)
        try:
            out[k.strip().upper()] = float(v.strip())
        except Exception:
            out[k.strip().upper()] = 0.0
    return out


def parse_bvh_floor_spec(value: str | None = None) -> Dict[str, int]:
    raw = parse_bvh_number_spec(value, DEFAULT_FLOOR_SPEC)
    return {k: max(0, int(round(v))) for k, v in raw.items()}


def parse_bvh_boost_spec(value: str | None = None) -> Dict[str, float]:
    return parse_bvh_number_spec(value, DEFAULT_BOOST_SPEC)


def bvh_id_from_candidate(candidate: Dict[str, Any]) -> str:
    return str(candidate.get("bvh_type", {}).get("BVHTypeID", "")).upper()


def bvh_id_from_suggestion(suggestion: Dict[str, Any]) -> str:
    return str(suggestion.get("BVHTypeID", "")).upper()


def boost_reason(candidate: Dict[str, Any], min_canonical_score: float) -> Tuple[bool, str]:
    canonical = safe_float(candidate.get("BVHCanonicalScore"))
    specific = safe_float(candidate.get("BVHSpecificEvidenceScore"))
    classifier = safe_float(candidate.get("BVHTypeClassifierScore"))
    pair_score = safe_float(candidate.get("PairFinalScore"), safe_float(candidate.get("FinalRAINScore")))
    penalty = safe_float(candidate.get("BVHConfusionPenalty"))
    margin = safe_float(candidate.get("BVHMarginScore"))

    if canonical >= min_canonical_score:
        return True, f"Canonical BVH score {canonical:.2f} >= floor {min_canonical_score:.2f}."
    if specific >= 75 and pair_score >= 65 and penalty <= 12:
        return True, "Strong BVH-specific evidence and pair score with low confusion penalty."
    if classifier >= 70 and pair_score >= 65 and penalty <= 12:
        return True, "Strong BVH classifier evidence and pair score with low confusion penalty."
    if margin >= 20 and pair_score >= 70 and penalty <= 18:
        return True, "Clear canonical margin and strong pair score."
    return False, (
        f"No recall boost: canonical={canonical:.2f}, specific={specific:.2f}, "
        f"classifier={classifier:.2f}, pair={pair_score:.2f}, penalty={penalty:.2f}."
    )


def apply_bvh_recall_boost(
    candidate: Dict[str, Any],
    *,
    boost_by_type: Dict[str, float],
    min_canonical_score: float = 50.0,
) -> Dict[str, Any]:
    bvh_id = bvh_id_from_candidate(candidate)
    boost = float(boost_by_type.get(bvh_id, 0.0))
    if boost <= 0:
        return {
            "BVHRecallBoost": 0.0,
            "BVHRecallBoostApplied": False,
            "BVHRecallBoostReason": "BVHType is not configured for recall boost.",
        }

    ok, reason = boost_reason(candidate, min_canonical_score)
    if not ok:
        return {
            "BVHRecallBoost": 0.0,
            "BVHRecallBoostApplied": False,
            "BVHRecallBoostReason": reason,
        }

    before = safe_float(candidate.get("FinalRAINScore"))
    before_sat = safe_float(candidate.get("FinalRAINScoreBeforeSaturation"), before)
    pair_final = safe_float(candidate.get("PairFinalScore"))

    candidate["FinalRAINScoreBeforeSaturation"] = round(before_sat + boost, 2)
    candidate["FinalRAINScore"] = round(before + boost, 2)
    candidate["PairFinalScore"] = round(pair_final + boost * 0.35, 2)

    return {
        "BVHRecallBoost": round(boost, 2),
        "BVHRecallBoostApplied": True,
        "BVHRecallBoostReason": reason,
        "FinalRAINScoreBeforeRecallBoost": round(before, 2),
    }


def count_by_bvh(suggestions: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for s in suggestions:
        bvh_id = bvh_id_from_suggestion(s)
        counts[bvh_id] = counts.get(bvh_id, 0) + 1
    return counts


def suggestion_pair_key(suggestion: Dict[str, Any]) -> Tuple[str, str]:
    return pair_key(suggestion.get("PersonAName", ""), suggestion.get("PersonBName", ""))


def pair_precision_pct(suggestions: List[Dict[str, Any]], calibration: Dict[str, Any]) -> float:
    positive_pairs = calibration.get("positive_pair_names", set())
    if not positive_pairs:
        return 100.0
    pred_pairs = {suggestion_pair_key(s) for s in suggestions}
    if not pred_pairs:
        return 0.0
    return len(pred_pairs & positive_pairs) / len(pred_pairs) * 100.0


def balance_bvh_type_floors(
    *,
    ranked_pool: List[Dict[str, Any]],
    selected: List[Dict[str, Any]],
    calibration: Dict[str, Any],
    floors_by_type: Dict[str, int],
    target_pair_precision_pct: float,
    max_suggestions: int,
    min_candidate_score: float = 0.0,
    min_bvh_canonical_score: float = 0.0,
    allow_precision_drop_pct: float = 0.0,
    use_backtest_precision_guard: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Add high-ranking canonical top-1 suggestions from underrepresented BVHTypes until
    configured type floors are met, without dropping pair precision below the target
    in backtest mode.

    This aims to recover soft BVHTypes such as PARTNERSHIP / TRUSTPATH / RECRUITMENT /
    SUPPLIERMATCH after auto-top-K has truncated the ranked list.
    """
    if not floors_by_type:
        return selected, {"enabled": False, "reason": "No BVH floors configured."}

    selected_out = list(selected)
    selected_keys = {
        (normalise_text(s.get("PersonAName", "")), normalise_text(s.get("PersonBName", "")), bvh_id_from_suggestion(s))
        for s in selected_out
    }

    before_counts = count_by_bvh(selected_out)
    added: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []

    precision_floor = max(0.0, float(target_pair_precision_pct) - float(allow_precision_drop_pct))

    for bvh_id, floor in floors_by_type.items():
        if floor <= 0:
            continue
        while count_by_bvh(selected_out).get(bvh_id, 0) < floor and len(selected_out) < max_suggestions:
            candidate_to_add = None
            for s in ranked_pool:
                if bvh_id_from_suggestion(s) != bvh_id:
                    continue
                key = (normalise_text(s.get("PersonAName", "")), normalise_text(s.get("PersonBName", "")), bvh_id_from_suggestion(s))
                if key in selected_keys:
                    continue
                if safe_float(s.get("FinalRAINScore")) < min_candidate_score:
                    continue
                if safe_float(s.get("BVHCanonicalScore")) < min_bvh_canonical_score:
                    continue
                candidate_to_add = s
                break

            if candidate_to_add is None:
                rejected.append({
                    "BVHTypeID": bvh_id,
                    "reason": "No candidate met score/canonical thresholds or capacity limit.",
                    "current_count": count_by_bvh(selected_out).get(bvh_id, 0),
                    "floor": floor,
                })
                break

            trial = selected_out + [candidate_to_add]
            trial_precision = pair_precision_pct(trial, calibration)
            if use_backtest_precision_guard and trial_precision < precision_floor:
                rejected.append({
                    "BVHTypeID": bvh_id,
                    "reason": "Precision guard rejected candidate.",
                    "candidate_pair": f"{candidate_to_add.get('PersonAName','')} <> {candidate_to_add.get('PersonBName','')}",
                    "candidate_score": safe_float(candidate_to_add.get("FinalRAINScore")),
                    "candidate_bvh_canonical_score": safe_float(candidate_to_add.get("BVHCanonicalScore")),
                    "trial_pair_precision_pct": round(trial_precision, 4),
                    "precision_floor_pct": round(precision_floor, 4),
                    "current_count": count_by_bvh(selected_out).get(bvh_id, 0),
                    "floor": floor,
                })
                selected_keys.add((normalise_text(candidate_to_add.get("PersonAName", "")), normalise_text(candidate_to_add.get("PersonBName", "")), bvh_id_from_suggestion(candidate_to_add)))
                continue

            candidate_to_add["BVHRecallBalancerAdded"] = True
            candidate_to_add["BVHRecallBalancerReason"] = f"Added to meet {bvh_id} floor {floor}."
            selected_out.append(candidate_to_add)
            selected_keys.add((normalise_text(candidate_to_add.get("PersonAName", "")), normalise_text(candidate_to_add.get("PersonBName", "")), bvh_id_from_suggestion(candidate_to_add)))
            added.append({
                "BVHTypeID": bvh_id,
                "PersonAName": candidate_to_add.get("PersonAName", ""),
                "PersonBName": candidate_to_add.get("PersonBName", ""),
                "FinalRAINScore": safe_float(candidate_to_add.get("FinalRAINScore")),
                "BVHCanonicalScore": safe_float(candidate_to_add.get("BVHCanonicalScore")),
                "PairPrecisionAfterAddPct": round(pair_precision_pct(selected_out, calibration), 4),
            })

    after_counts = count_by_bvh(selected_out)
    selected_out.sort(
        key=lambda x: (
            safe_float(x.get("FinalRAINScore")),
            safe_float(x.get("PairFinalScore")),
            safe_float(x.get("BVHCanonicalScore")),
            safe_float(x.get("PairPrototypeScore")),
            safe_float(x.get("SpecificPairOverlapScore")),
        ),
        reverse=True,
    )

    return selected_out, {
        "enabled": True,
        "floors_by_type": floors_by_type,
        "before_counts": before_counts,
        "after_counts": after_counts,
        "target_pair_precision_pct": target_pair_precision_pct,
        "precision_floor_pct": precision_floor,
        "use_backtest_precision_guard": use_backtest_precision_guard,
        "min_candidate_score": min_candidate_score,
        "min_bvh_canonical_score": min_bvh_canonical_score,
        "added_count": len(added),
        "added": added,
        "rejected_count": len(rejected),
        "rejected": rejected[:100],
    }
