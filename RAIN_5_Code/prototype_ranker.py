from __future__ import annotations
from typing import Any, Dict, List, Tuple, Set
import csv
import json
import math
from pathlib import Path

from .calibration import pair_key, pair_bvh_key
from .utils import normalise_text, token_set, safe_float, write_json
from .supervised_ranker import candidate_features


STOPWORDS = {
    "the", "and", "for", "with", "from", "that", "this", "into", "onto", "than", "then",
    "have", "has", "had", "will", "would", "could", "should", "about", "across", "within",
    "their", "there", "where", "when", "what", "which", "business", "company", "client",
    "clients", "market", "markets", "solution", "solutions", "service", "services", "need",
    "needs", "value", "commercial", "strategic", "strategy", "growth", "support", "help",
    "team", "teams", "senior", "director", "manager", "head", "lead", "leading", "global",
    "local", "new", "existing", "strong", "high", "low", "large", "small", "data",
}


DOMINANT_FALSE_POSITIVE_BVHS = {
    "PARTNERSHIP",
    "PRODUCTSALE",
    "SUPPLIERMATCH",
    "PROBLEMSOLUTION",
}

_PROFILE_TOKEN_VIEW_CACHE: Dict[str, Dict[str, Set[str]]] = {}


def profile_cache_key(profile: Dict[str, Any]) -> str:
    return str(profile.get("PersonID") or profile.get("ProfileIndex") or profile.get("full_name") or id(profile))


def clean_tokens(value: Any) -> Set[str]:
    return {t for t in token_set(value) if len(t) >= 3 and t not in STOPWORDS}


def tokens_from_fields(profile: Dict[str, Any], fields: List[str]) -> Set[str]:
    out: Set[str] = set()
    for field in fields:
        value = profile.get(field, "")
        if isinstance(value, (dict, list)):
            value = json.dumps(value, ensure_ascii=False)
        out |= clean_tokens(value)
    return out


def profile_token_views(profile: Dict[str, Any]) -> Dict[str, Set[str]]:
    cache_key = profile_cache_key(profile)
    cached = _PROFILE_TOKEN_VIEW_CACHE.get(cache_key)
    if cached is not None:
        return cached

    supply_fields = [
        "offers_json",
        "solutions_json",
        "product_value_cases_json",
        "use_cases_json",
        "credibility_statements_json",
        "commercial_pricing_json",
    ]
    demand_fields = [
        "needs_json",
        "buyer_pain_points_json",
        "use_cases_json",
        "regulatory_drivers_json",
        "commercial_context_json",
    ]
    insight_fields = [
        "insights_json",
        "regulatory_drivers_json",
        "commercial_context_json",
        "primary_driver_description",
        "secondary_driver_description",
    ]
    role_fields = [
        "current_role",
        "organisation_name",
        "organisation_archetype",
        "seniority",
        "role_family",
        "persona_cluster",
        "market_regime",
        "primary_driver_description",
        "secondary_driver_description",
    ]
    search_text = profile.get("_search_text", "")
    views = {
        "all": clean_tokens(search_text),
        "supply": tokens_from_fields(profile, supply_fields),
        "demand": tokens_from_fields(profile, demand_fields),
        "insight": tokens_from_fields(profile, insight_fields),
        "role_context": tokens_from_fields(profile, role_fields),
    }
    _PROFILE_TOKEN_VIEW_CACHE[cache_key] = views
    return views


def jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def overlap_coeff(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, min(len(a), len(b)))


def normalised_name(name: Any) -> str:
    return normalise_text(name)


def build_profile_name_index(profiles: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for p in profiles:
        name = normalised_name(p.get("full_name", ""))
        if name:
            index[name] = p
    return index


def pair_signature(profile_a: Dict[str, Any], profile_b: Dict[str, Any]) -> Dict[str, Any]:
    a = profile_token_views(profile_a)
    b = profile_token_views(profile_b)

    return {
        "a_name": profile_a.get("full_name", ""),
        "b_name": profile_b.get("full_name", ""),
        "a_supply": a["supply"],
        "a_demand": a["demand"],
        "a_insight": a["insight"],
        "a_context": a["all"] | a["role_context"],
        "b_supply": b["supply"],
        "b_demand": b["demand"],
        "b_insight": b["insight"],
        "b_context": b["all"] | b["role_context"],
        "pair_context": a["all"] | b["all"] | a["role_context"] | b["role_context"],
        "pair_supply": a["supply"] | b["supply"],
        "pair_demand": a["demand"] | b["demand"],
        "pair_insight": a["insight"] | b["insight"],
        "primary_drivers": {
            normalise_text(profile_a.get("primary_driver_id") or profile_a.get("primary_driver_description")),
            normalise_text(profile_b.get("primary_driver_id") or profile_b.get("primary_driver_description")),
        },
        "market_regimes": {
            normalise_text(profile_a.get("market_regime")),
            normalise_text(profile_b.get("market_regime")),
        },
        "role_families": {
            normalise_text(profile_a.get("role_family")),
            normalise_text(profile_b.get("role_family")),
        },
    }


def build_pair_prototype_model(
    profiles: List[Dict[str, Any]],
    calibration: Dict[str, Any],
    max_prototypes: int = 1000,
) -> Dict[str, Any]:
    profile_index = build_profile_name_index(profiles)
    prototypes: List[Dict[str, Any]] = []

    positives = list(calibration.get("positive_examples", []) or [])
    positives.sort(
        key=lambda ex: safe_float(ex.get("HVTI.ExpectedLifetimeValueGBP"), safe_float(ex.get("HVTI.PotentialLifetimeValueGBP"))),
        reverse=True,
    )

    for ex in positives[:max_prototypes]:
        a = profile_index.get(normalised_name(ex.get("PersonAName", "")))
        b = profile_index.get(normalised_name(ex.get("PersonBName", "")))
        if not a or not b:
            continue

        sig = pair_signature(a, b)
        prototypes.append({
            "person_a": ex.get("PersonAName", ""),
            "person_b": ex.get("PersonBName", ""),
            "bvh_type_id": normalise_text(ex.get("BVHTypeID", "")).replace(" ", "").upper(),
            "bvh_type_name": ex.get("BVHTypeName", ""),
            "expected_value_gbp": safe_float(ex.get("HVTI.ExpectedLifetimeValueGBP"), safe_float(ex.get("HVTI.PotentialLifetimeValueGBP"))),
            "signature": sig,
        })

    return {
        "enabled": bool(prototypes),
        "method": "positive_label_pair_prototype_nearest_neighbour",
        "prototype_count": len(prototypes),
        "prototypes": prototypes,
    }



def serialise_pair_prototype_model(model: Dict[str, Any]) -> Dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, set):
            return sorted(value)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        if isinstance(value, list):
            return [convert(v) for v in value]
        return value
    return convert(model)


def score_specific_pair_overlap(candidate: Dict[str, Any]) -> Dict[str, Any]:
    source = candidate.get("source", {})
    target = candidate.get("target", {})
    s = profile_token_views(source)
    t = profile_token_views(target)

    source_supply_to_target_demand = overlap_coeff(s["supply"], t["demand"])
    target_supply_to_source_demand = overlap_coeff(t["supply"], s["demand"])
    source_insight_to_target_demand = overlap_coeff(s["insight"], t["demand"])
    target_insight_to_source_demand = overlap_coeff(t["insight"], s["demand"])
    context_overlap = jaccard(s["all"] | s["role_context"], t["all"] | t["role_context"])

    same_driver = 1.0 if normalise_text(source.get("primary_driver_id") or source.get("primary_driver_description")) == normalise_text(target.get("primary_driver_id") or target.get("primary_driver_description")) else 0.0
    same_regime = 1.0 if normalise_text(source.get("market_regime")) == normalise_text(target.get("market_regime")) else 0.0

    best_supply_demand = max(source_supply_to_target_demand, target_supply_to_source_demand)
    best_insight_demand = max(source_insight_to_target_demand, target_insight_to_source_demand)
    alignment = max(same_driver, same_regime)

    score = (
        55.0 * best_supply_demand
        + 20.0 * best_insight_demand
        + 15.0 * context_overlap
        + 10.0 * alignment
    )

    return {
        "SpecificPairOverlapScore": round(min(100.0, max(0.0, score)), 2),
        "SpecificPairOverlapJSON": {
            "source_supply_to_target_demand": round(source_supply_to_target_demand, 6),
            "target_supply_to_source_demand": round(target_supply_to_source_demand, 6),
            "source_insight_to_target_demand": round(source_insight_to_target_demand, 6),
            "target_insight_to_source_demand": round(target_insight_to_source_demand, 6),
            "context_overlap": round(context_overlap, 6),
            "same_driver": same_driver,
            "same_regime": same_regime,
            "source_supply_target_demand_intersection": sorted((s["supply"] & t["demand"]))[:30],
            "target_supply_source_demand_intersection": sorted((t["supply"] & s["demand"]))[:30],
        },
    }


def score_against_prototypes(candidate: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    if not model or not model.get("enabled"):
        return {
            "PairPrototypeScore": 0.0,
            "PairPrototypeMatch": "",
            "PairPrototypeMatchBVHTypeID": "",
            "PairPrototypeSimilarityJSON": {},
        }

    source = candidate.get("source", {})
    target = candidate.get("target", {})
    cand = pair_signature(source, target)
    cand_bvh = normalise_text(candidate.get("bvh_type", {}).get("BVHTypeID", "")).replace(" ", "").upper()

    best_score = 0.0
    best_proto = None
    best_trace: Dict[str, Any] = {}

    for proto in model.get("prototypes", []):
        psig = proto.get("signature", {})

        context_sim = jaccard(cand["pair_context"], psig.get("pair_context", set()))
        supply_sim = jaccard(cand["pair_supply"], psig.get("pair_supply", set()))
        demand_sim = jaccard(cand["pair_demand"], psig.get("pair_demand", set()))
        insight_sim = jaccard(cand["pair_insight"], psig.get("pair_insight", set()))

        driver_sim = jaccard({x for x in cand["primary_drivers"] if x}, {x for x in psig.get("primary_drivers", set()) if x})
        regime_sim = jaccard({x for x in cand["market_regimes"] if x}, {x for x in psig.get("market_regimes", set()) if x})
        role_sim = jaccard({x for x in cand["role_families"] if x}, {x for x in psig.get("role_families", set()) if x})
        bvh_match = 1.0 if cand_bvh and cand_bvh == proto.get("bvh_type_id") else 0.0

        score = (
            30.0 * context_sim
            + 20.0 * max(supply_sim, demand_sim)
            + 15.0 * insight_sim
            + 15.0 * max(driver_sim, regime_sim)
            + 10.0 * role_sim
            + 10.0 * bvh_match
        )

        if score > best_score:
            best_score = score
            best_proto = proto
            best_trace = {
                "context_similarity": round(context_sim, 6),
                "supply_similarity": round(supply_sim, 6),
                "demand_similarity": round(demand_sim, 6),
                "insight_similarity": round(insight_sim, 6),
                "driver_similarity": round(driver_sim, 6),
                "regime_similarity": round(regime_sim, 6),
                "role_similarity": round(role_sim, 6),
                "bvh_match": bvh_match,
            }

    match_name = ""
    match_bvh = ""
    if best_proto:
        match_name = f"{best_proto.get('person_a', '')} <> {best_proto.get('person_b', '')}"
        match_bvh = best_proto.get("bvh_type_id", "")

    return {
        "PairPrototypeScore": round(min(100.0, max(0.0, best_score)), 2),
        "PairPrototypeMatch": match_name,
        "PairPrototypeMatchBVHTypeID": match_bvh,
        "PairPrototypeSimilarityJSON": best_trace,
    }


def compute_generic_pattern_penalty(
    candidate: Dict[str, Any],
    features: Dict[str, float],
    pair_prototype_score: float,
    specific_pair_overlap_score: float,
) -> Dict[str, Any]:
    bvh_id = normalise_text(candidate.get("bvh_type", {}).get("BVHTypeID", "")).replace(" ", "").upper()

    generic_flags = [
        features.get("source_has_product_or_solution", 0.0) >= 1.0,
        features.get("target_has_need_or_pain", 0.0) >= 1.0,
        features.get("source_has_network_access", 0.0) >= 1.0,
        features.get("source_has_insight", 0.0) >= 1.0,
        features.get("same_primary_driver", 0.0) >= 1.0,
        features.get("same_market_regime", 0.0) >= 1.0,
        features.get("product_need_interaction", 0.0) >= 1.0,
    ]
    generic_count = sum(1 for x in generic_flags if x)
    penalty = 0.0
    reasons: List[str] = []

    if generic_count >= 5 and specific_pair_overlap_score < 25 and pair_prototype_score < 35:
        penalty += 35.0
        reasons.append("Generic evidence pattern without specific pair overlap/prototype fit.")
    elif generic_count >= 5 and specific_pair_overlap_score < 40:
        penalty += 20.0
        reasons.append("Generic evidence pattern with weak specific overlap.")
    elif generic_count >= 4 and specific_pair_overlap_score < 25:
        penalty += 12.0
        reasons.append("Broad commercial evidence but low specific overlap.")

    if bvh_id in DOMINANT_FALSE_POSITIVE_BVHS and specific_pair_overlap_score < 35:
        penalty += 12.0
        reasons.append(f"{bvh_id} was dominant in false positives and has weak specific overlap.")

    if pair_prototype_score < 20 and specific_pair_overlap_score < 20:
        penalty += 12.0
        reasons.append("Low prototype and low specific-overlap support.")

    return {
        "GenericPatternPenalty": round(min(60.0, penalty), 2),
        "GenericPatternScore": round(generic_count / max(1, len(generic_flags)) * 100.0, 2),
        "GenericPatternReason": " | ".join(reasons) if reasons else "No generic-pattern penalty.",
    }


def apply_prototype_antigeneric_scores(candidate: Dict[str, Any], prototype_model: Dict[str, Any]) -> Dict[str, Any]:
    features = candidate_features(candidate)
    proto = score_against_prototypes(candidate, prototype_model)
    specific = score_specific_pair_overlap(candidate)
    generic = compute_generic_pattern_penalty(
        candidate=candidate,
        features=features,
        pair_prototype_score=safe_float(proto.get("PairPrototypeScore")),
        specific_pair_overlap_score=safe_float(specific.get("SpecificPairOverlapScore")),
    )

    value_score = safe_float(candidate.get("component_scores", {}).get("ValueScore"))
    pair_score = safe_float(candidate.get("PairSupervisedScore"))
    bvh_score = safe_float(candidate.get("BVHTypeSupervisedScore"))
    hard_gate = safe_float(candidate.get("HardBVHGateMultiplier"), 1.0)

    raw = (
        0.35 * safe_float(proto.get("PairPrototypeScore"))
        + 0.25 * safe_float(specific.get("SpecificPairOverlapScore"))
        + 0.20 * pair_score
        + 0.10 * bvh_score
        + 0.10 * value_score
    )
    before_saturation = max(0.0, raw * hard_gate - safe_float(generic.get("GenericPatternPenalty")))

    trace = candidate.get("SupervisedFeatureTraceJSON", {}) or {}
    if not isinstance(trace, dict):
        trace = {}
    trace["prototype_antigeneric"] = {
        "pair_prototype": proto,
        "specific_pair_overlap": specific,
        "generic_pattern_penalty": generic,
        "formula": "FinalRAINScore = (0.35*PairPrototypeScore + 0.25*SpecificPairOverlapScore + 0.20*PairSupervisedScore + 0.10*BVHTypeSupervisedScore + 0.10*ValueScore) * HardBVHGateMultiplier - GenericPatternPenalty - PersonSaturationPenalty",
    }

    return {
        **proto,
        **specific,
        **generic,
        "FinalRAINScorePrototypeRaw": round(raw, 2),
        "FinalRAINScoreBeforeSaturation": round(before_saturation, 2),
        "PersonSaturationPenalty": 0.0,
        "FinalRAINScore": round(before_saturation, 2),
        "SupervisedFeatureTraceJSON": trace,
    }


def suggestion_people(s: Dict[str, Any]) -> Tuple[str, str]:
    return normalise_text(s.get("PersonAName", "")), normalise_text(s.get("PersonBName", ""))


def apply_person_saturation_penalty_to_suggestions(
    suggestions: List[Dict[str, Any]],
    high_score_threshold: float = 70.0,
    weak_specific_threshold: float = 35.0,
    penalty_per_extra: float = 4.0,
    max_penalty: float = 35.0,
) -> Dict[str, Any]:
    """
    Penalise people who appear in many high-scoring but generic/weak-specific alternatives.
    This attacks the observed failure mode: "this person looks good with everyone."
    """
    counts: Dict[str, int] = {}
    for s in suggestions:
        score = safe_float(s.get("FinalRAINScoreBeforeSaturation"), safe_float(s.get("FinalRAINScore")))
        specific = safe_float(s.get("SpecificPairOverlapScore"))
        generic_penalty = safe_float(s.get("GenericPatternPenalty"))
        if score >= high_score_threshold and (specific < weak_specific_threshold or generic_penalty > 0):
            a, b = suggestion_people(s)
            counts[a] = counts.get(a, 0) + 1
            counts[b] = counts.get(b, 0) + 1

    total_penalty = 0.0
    penalised_count = 0
    for s in suggestions:
        a, b = suggestion_people(s)
        exposure = max(counts.get(a, 0), counts.get(b, 0))
        penalty = min(max_penalty, max(0, exposure - 3) * penalty_per_extra)
        before = safe_float(s.get("FinalRAINScoreBeforeSaturation"), safe_float(s.get("FinalRAINScore")))
        final = max(0.0, before - penalty)

        s["PersonSaturationPenalty"] = round(penalty, 2)
        s["FinalRAINScore"] = round(final, 2)

        candidate = s.get("_candidate")
        if isinstance(candidate, dict):
            candidate["PersonSaturationPenalty"] = round(penalty, 2)
            candidate["FinalRAINScore"] = round(final, 2)

        if penalty > 0:
            penalised_count += 1
            total_penalty += penalty

    return {
        "enabled": True,
        "high_score_threshold": high_score_threshold,
        "weak_specific_threshold": weak_specific_threshold,
        "penalty_per_extra": penalty_per_extra,
        "max_penalty": max_penalty,
        "person_generic_exposure_counts": dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:50]),
        "penalised_suggestion_count": penalised_count,
        "total_penalty": round(total_penalty, 2),
    }


def candidate_rank_row(candidate: Dict[str, Any], keep_candidate: bool) -> Dict[str, Any]:
    source = candidate.get("source", {})
    target = candidate.get("target", {})
    bvh = candidate.get("bvh_type", {})
    pk = pair_key(source.get("full_name", ""), target.get("full_name", ""))
    pbk = pair_bvh_key(source.get("full_name", ""), target.get("full_name", ""), bvh.get("BVHTypeID", ""))
    return {
        "pair_key": pk,
        "pair_bvh_key": pbk,
        "person_a": source.get("full_name", ""),
        "person_b": target.get("full_name", ""),
        "bvh_type_id": bvh.get("BVHTypeID", ""),
        "bvh_type_name": bvh.get("BVHTypeName", ""),
        "passed_filter": bool(keep_candidate),
        "estimated_hvti_score": safe_float(candidate.get("hvti_result", {}).get("EstimatedHVTIScore")),
        "estimated_usd_value": safe_float(candidate.get("roi_result", {}).get("EstimatedHVITUSDValue")),
        "pair_supervised_score": safe_float(candidate.get("PairSupervisedScore")),
        "bvh_type_supervised_score": safe_float(candidate.get("BVHTypeSupervisedScore")),
        "pair_prototype_score": safe_float(candidate.get("PairPrototypeScore")),
        "specific_pair_overlap_score": safe_float(candidate.get("SpecificPairOverlapScore")),
        "generic_pattern_penalty": safe_float(candidate.get("GenericPatternPenalty")),
        "hard_bvh_gate_pass": candidate.get("HardBVHGatePass", True),
        "hard_bvh_gate_reason": candidate.get("HardBVHGateReason", ""),
        "final_rain_score_before_saturation": safe_float(candidate.get("FinalRAINScoreBeforeSaturation"), safe_float(candidate.get("FinalRAINScore"))),
        "final_rain_score": safe_float(candidate.get("FinalRAINScore")),
        "pair_prototype_match": candidate.get("PairPrototypeMatch", ""),
        "pair_prototype_match_bvh_type_id": candidate.get("PairPrototypeMatchBVHTypeID", ""),
        "generic_pattern_reason": candidate.get("GenericPatternReason", ""),
    }


def rank_rows_by_score(rank_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        rank_rows,
        key=lambda r: (
            safe_float(r.get("final_rain_score")),
            safe_float(r.get("pair_prototype_score")),
            safe_float(r.get("specific_pair_overlap_score")),
            safe_float(r.get("pair_supervised_score")),
        ),
        reverse=True,
    )


def build_label_coverage_report(
    calibration: Dict[str, Any],
    rank_rows: List[Dict[str, Any]],
    final_suggestions: List[Dict[str, Any]],
    max_suggestions: int,
) -> List[Dict[str, Any]]:
    ranked = rank_rows_by_score(rank_rows)
    pair_best: Dict[Tuple[str, str], Dict[str, Any]] = {}
    pair_bvh_best: Dict[Tuple[Tuple[str, str], str], Dict[str, Any]] = {}

    for idx, row in enumerate(ranked, 1):
        row = dict(row)
        row["raw_rank"] = idx
        pk = row["pair_key"]
        pbk = row["pair_bvh_key"]
        if pk not in pair_best:
            pair_best[pk] = row
        if pbk not in pair_bvh_best:
            pair_bvh_best[pbk] = row

    suggested_pairs = {pair_key(s.get("PersonAName", ""), s.get("PersonBName", "")) for s in final_suggestions}
    suggested_pair_bvhs = {
        pair_bvh_key(s.get("PersonAName", ""), s.get("PersonBName", ""), s.get("BVHTypeID", ""))
        for s in final_suggestions
    }

    rows: List[Dict[str, Any]] = []
    for ex in calibration.get("positive_examples", []) or []:
        pk = ex.get("pair_key") or pair_key(ex.get("PersonAName", ""), ex.get("PersonBName", ""))
        pbk = ex.get("pair_bvh_key") or pair_bvh_key(ex.get("PersonAName", ""), ex.get("PersonBName", ""), ex.get("BVHTypeID", ""))
        pair_row = pair_best.get(pk, {})
        bvh_row = pair_bvh_best.get(pbk, {})

        pair_found = pk in suggested_pairs
        bvh_found = pbk in suggested_pair_bvhs

        if not pair_row:
            reason = "NOT_GENERATED"
        elif not pair_row.get("passed_filter"):
            reason = "FILTERED_OUT_BY_MIN_SCORE_OR_VALUE"
        elif not pair_found:
            reason = "RANKED_BELOW_OUTPUT_TOP_K"
        elif pair_found and not bvh_found:
            reason = "PAIR_FOUND_WRONG_BVH"
        elif bvh_found:
            reason = "FOUND_PAIR_AND_BVH"
        else:
            reason = "UNKNOWN"

        if bvh_row and bvh_row.get("hard_bvh_gate_pass") is False:
            reason = "HARD_BVH_GATE_PENALISED"

        rows.append({
            "PersonAName": ex.get("PersonAName", ""),
            "PersonBName": ex.get("PersonBName", ""),
            "ExpectedBVHTypeID": ex.get("BVHTypeID", ""),
            "ExpectedBVHTypeName": ex.get("BVHTypeName", ""),
            "ExpectedValueGBP": safe_float(ex.get("HVTI.ExpectedLifetimeValueGBP"), safe_float(ex.get("HVTI.PotentialLifetimeValueGBP"))),
            "PairFoundInFinalSuggestions": pair_found,
            "PairBVHFoundInFinalSuggestions": bvh_found,
            "BestPairRawRank": pair_row.get("raw_rank", ""),
            "BestPairBVHRawRank": bvh_row.get("raw_rank", ""),
            "BestPairPassedFilter": pair_row.get("passed_filter", ""),
            "BestPairBVHPassedFilter": bvh_row.get("passed_filter", ""),
            "BestPairBVHTypeID": pair_row.get("bvh_type_id", ""),
            "ExpectedPairBVHTypeID": bvh_row.get("bvh_type_id", ""),
            "BestPairFinalRAINScore": pair_row.get("final_rain_score", ""),
            "BestPairBVHFinalRAINScore": bvh_row.get("final_rain_score", ""),
            "PairSupervisedScore": pair_row.get("pair_supervised_score", ""),
            "BVHTypeSupervisedScore": bvh_row.get("bvh_type_supervised_score", ""),
            "PairPrototypeScore": pair_row.get("pair_prototype_score", ""),
            "SpecificPairOverlapScore": pair_row.get("specific_pair_overlap_score", ""),
            "GenericPatternPenalty": pair_row.get("generic_pattern_penalty", ""),
            "HardBVHGatePass": bvh_row.get("hard_bvh_gate_pass", ""),
            "HardBVHGateReason": bvh_row.get("hard_bvh_gate_reason", ""),
            "PairPrototypeMatch": pair_row.get("pair_prototype_match", ""),
            "PairPrototypeMatchBVHTypeID": pair_row.get("pair_prototype_match_bvh_type_id", ""),
            "GenericPatternReason": pair_row.get("generic_pattern_reason", ""),
            "ReasonMissed": reason,
            "ScoreGapToTop50": "",
        })

    return rows


def write_csv_rows(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            f.write("")
        return
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                fieldnames.append(k)
                seen.add(k)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_label_coverage_report(output_run_dir: Path, rows: List[Dict[str, Any]]) -> Dict[str, str]:
    csv_path = output_run_dir / "LabelCoverageReport.csv"
    json_path = output_run_dir / "LabelCoverageReport.JSON"
    write_csv_rows(csv_path, rows)
    write_json(json_path, {"schema_version": "1.0", "rows": rows})
    return {"csv": str(csv_path), "json": str(json_path)}
