from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional
import json
import math

from .utils import normalise_text, safe_float, token_set, write_json
from .calibration import pair_key, pair_bvh_key
from .supervised_ranker import (
    train_logistic_ranker,
    predict_label_probability,
    top_feature_contributions,
    candidate_features,
)


CONFUSION_CLUSTER_PRODUCT = {"PRODUCTSALE", "PROBLEMSOLUTION", "SUPPLIERMATCH", "PARTNERSHIP"}
NETWORK_BVHS = {"CLIENTINTRODUCTION", "CAPITALINTRODUCTION", "ECOSYSTEMAMPLIFICATION", "TRUSTPATH"}
KNOWN_BVHS = [
    "PRODUCTSALE",
    "CLIENTINTRODUCTION",
    "PROBLEMSOLUTION",
    "INSIGHTEXCHANGE",
    "CAPITALINTRODUCTION",
    "PARTNERSHIP",
    "RECRUITMENT",
    "SUPPLIERMATCH",
    "TRUSTPATH",
    "ECOSYSTEMAMPLIFICATION",
]

CAPITAL_TERMS = {"capital", "fund", "funding", "investor", "investment", "debt", "equity", "credit", "lender", "raise"}
SUPPLIER_TERMS = {"supplier", "vendor", "procure", "procurement", "supply", "outsourcing", "service", "delivery"}
PARTNERSHIP_TERMS = {"partner", "partnership", "joint", "alliance", "strategic", "collaboration", "co-create", "ecosystem"}
CLIENT_TERMS = {"client", "customer", "buyer", "route", "market", "intro", "introduce", "relationship"}
TRUST_TERMS = {"trust", "trusted", "credibility", "reference", "reputation", "advisor", "warm"}
INSIGHT_TERMS = {"insight", "regulation", "market", "research", "expert", "knowledge", "thought", "advice"}
HIRING_TERMS = {"hire", "hiring", "recruit", "recruitment", "talent", "candidate", "workforce", "job", "people"}


def sigmoid(x: float) -> float:
    if x < -40:
        return 0.0
    if x > 40:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def text_tokens(profile: Dict[str, Any]) -> set:
    text = " ".join([
        str(profile.get("_search_text", "")),
        str(profile.get("current_role", "")),
        str(profile.get("role_family", "")),
        str(profile.get("organisation_archetype", "")),
        json.dumps(profile.get("offers_json", ""), ensure_ascii=False),
        json.dumps(profile.get("needs_json", ""), ensure_ascii=False),
        json.dumps(profile.get("solutions_json", ""), ensure_ascii=False),
        json.dumps(profile.get("insights_json", ""), ensure_ascii=False),
        json.dumps(profile.get("commercial_context_json", ""), ensure_ascii=False),
        json.dumps(profile.get("buyer_pain_points_json", ""), ensure_ascii=False),
        json.dumps(profile.get("relationship_edges_json", ""), ensure_ascii=False),
    ])
    return token_set(text)


def has_any(tokens: set, terms: set) -> float:
    return 1.0 if tokens & terms else 0.0


def enhance_bvh_features(base: Dict[str, float], candidate: Dict[str, Any]) -> Dict[str, float]:
    f = dict(base)
    source = candidate.get("source", {})
    target = candidate.get("target", {})
    st = text_tokens(source)
    tt = text_tokens(target)

    f["pair_prototype_score"] = safe_float(candidate.get("PairPrototypeScore")) / 100.0
    f["specific_pair_overlap_score"] = safe_float(candidate.get("SpecificPairOverlapScore")) / 100.0
    f["generic_pattern_penalty_norm"] = safe_float(candidate.get("GenericPatternPenalty")) / 100.0
    f["bvh_type_supervised_score_norm"] = safe_float(candidate.get("BVHTypeSupervisedScore")) / 100.0
    f["pair_supervised_score_norm"] = safe_float(candidate.get("PairSupervisedScore")) / 100.0
    f["hard_gate_score_norm"] = safe_float(candidate.get("HardBVHGateScore"), 100.0) / 100.0

    bvh_id = str(candidate.get("bvh_type", {}).get("BVHTypeID", "")).upper()
    proto_type = str(candidate.get("PairPrototypeMatchBVHTypeID", "")).upper()
    f["prototype_bvh_exact_match"] = 1.0 if proto_type and proto_type == bvh_id else 0.0
    f["prototype_bvh_mismatch"] = 1.0 if proto_type and proto_type != bvh_id else 0.0

    f["source_has_capital_terms"] = has_any(st, CAPITAL_TERMS)
    f["target_has_capital_terms"] = has_any(tt, CAPITAL_TERMS)
    f["source_has_supplier_terms"] = has_any(st, SUPPLIER_TERMS)
    f["target_has_supplier_terms"] = has_any(tt, SUPPLIER_TERMS)
    f["source_has_partnership_terms"] = has_any(st, PARTNERSHIP_TERMS)
    f["target_has_partnership_terms"] = has_any(tt, PARTNERSHIP_TERMS)
    f["source_has_client_terms"] = has_any(st, CLIENT_TERMS)
    f["target_has_client_terms"] = has_any(tt, CLIENT_TERMS)
    f["source_has_trust_terms_precise"] = has_any(st, TRUST_TERMS)
    f["target_has_trust_terms_precise"] = has_any(tt, TRUST_TERMS)
    f["source_has_insight_terms_precise"] = has_any(st, INSIGHT_TERMS)
    f["target_has_insight_terms_precise"] = has_any(tt, INSIGHT_TERMS)
    f["source_has_hiring_terms_precise"] = has_any(st, HIRING_TERMS)
    f["target_has_hiring_terms_precise"] = has_any(tt, HIRING_TERMS)

    f["capital_intro_signal"] = max(f["source_has_capital_terms"], f["target_has_capital_terms"]) * f.get("source_has_network_access", 0.0)
    f["supplier_match_signal"] = max(f["source_has_supplier_terms"], f["target_has_supplier_terms"]) * f.get("product_need_interaction", 0.0)
    f["client_intro_signal"] = max(f["source_has_client_terms"], f["target_has_client_terms"]) * f.get("source_has_network_access", 0.0)
    f["partnership_signal"] = max(f["source_has_partnership_terms"], f["target_has_partnership_terms"]) * max(f.get("capability_complementarity", 0.0), f.get("strategic_alignment", 0.0))
    f["trust_path_signal"] = max(f["source_has_trust_terms_precise"], f["target_has_trust_terms_precise"]) * f.get("trust_potential", 0.0)
    f["insight_signal"] = max(f["source_has_insight_terms_precise"], f["target_has_insight_terms_precise"], f.get("source_has_insight", 0.0))
    f["recruitment_signal"] = max(f["source_has_hiring_terms_precise"], f["target_has_hiring_terms_precise"])
    return f


def train_bvh_precision_models(
    training_rows: List[Dict[str, Any]],
    bvh_type_ids: List[str],
    *,
    negative_samples_per_positive: int = 10,
    epochs: int = 80,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    max_training_rows: int = 5000,
    random_seed: int = 17,
    high_value_positive_weight: float = 2.5,
) -> Dict[str, Any]:
    """
    Train one candidate-correctness classifier per BVHType.

    This is not a generic pair model. For PRODUCTSALE, for example, it trains
    only on PRODUCTSALE candidates and learns PRODUCTSALE-correct vs
    PRODUCTSALE-false-positive.
    """
    models: Dict[str, Any] = {}
    diagnostics: Dict[str, Any] = {}
    enabled_count = 0

    for bvh_id in [str(x).upper() for x in bvh_type_ids]:
        rows_for_type: List[Dict[str, Any]] = []
        for row in training_rows:
            row_bvh = str(row.get("BVHTypeID", "")).upper()
            if row_bvh != bvh_id:
                continue
            rr = dict(row)
            rr["Label"] = int(row.get("BVHTypeLabel", row.get("Label", 0)))
            rr["features"] = dict(row.get("bvh_features", row.get("features", {})))
            rows_for_type.append(rr)

        model = train_logistic_ranker(
            rows_for_type,
            negative_samples_per_positive=negative_samples_per_positive,
            epochs=epochs,
            learning_rate=learning_rate,
            l2=l2,
            max_training_rows=max_training_rows,
            random_seed=random_seed + abs(hash(bvh_id)) % 100000,
            high_value_positive_weight=high_value_positive_weight,
            model_name=f"one_vs_rest_bvh_type_{bvh_id}",
        )
        models[bvh_id] = model
        if model.get("enabled"):
            enabled_count += 1
        diagnostics[bvh_id] = {
            "enabled": model.get("enabled", False),
            "training_row_count_all": model.get("training_row_count_all", len(rows_for_type)),
            "positive_count_all": model.get("positive_count_all", sum(1 for r in rows_for_type if int(r.get("Label", 0)) == 1)),
            "negative_count_all": model.get("negative_count_all", sum(1 for r in rows_for_type if int(r.get("Label", 0)) == 0)),
            "diagnostics": model.get("diagnostics", {}),
            "reason": model.get("reason", ""),
        }

    return {
        "enabled": enabled_count > 0,
        "method": "one_classifier_per_bvh_type_candidate_correctness",
        "bvh_type_count": len(bvh_type_ids),
        "enabled_classifier_count": enabled_count,
        "models": models,
        "diagnostics": diagnostics,
    }


def bvh_specific_evidence_score(candidate: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
    bvh_id = str(candidate.get("bvh_type", {}).get("BVHTypeID", "")).upper()

    specific = safe_float(candidate.get("SpecificPairOverlapScore")) / 100.0
    pair_proto = safe_float(candidate.get("PairPrototypeScore")) / 100.0
    product_need = safe_float(features.get("product_need_interaction"))
    network = safe_float(features.get("source_has_network_access"))
    insight = max(safe_float(features.get("source_has_insight")), safe_float(features.get("insight_signal")))
    strategic = safe_float(features.get("strategic_alignment"))
    complement = safe_float(features.get("capability_complementarity"))
    trust = safe_float(features.get("trust_potential"))
    problem_fit = safe_float(features.get("problem_solution_fit"))
    network_amp = safe_float(features.get("network_amplification"))

    if bvh_id == "PRODUCTSALE":
        score = 45 * product_need + 25 * specific + 15 * safe_float(features.get("target_has_buying_terms")) + 15 * pair_proto
    elif bvh_id == "PROBLEMSOLUTION":
        score = 35 * problem_fit + 30 * product_need + 20 * specific + 15 * insight
    elif bvh_id == "SUPPLIERMATCH":
        score = 40 * safe_float(features.get("supplier_match_signal")) + 25 * product_need + 20 * specific + 15 * safe_float(features.get("target_has_supplier_terms"))
    elif bvh_id == "PARTNERSHIP":
        score = 35 * safe_float(features.get("partnership_signal")) + 25 * complement + 20 * strategic + 20 * specific
    elif bvh_id == "CLIENTINTRODUCTION":
        score = 45 * safe_float(features.get("client_intro_signal")) + 25 * network + 20 * network_amp + 10 * specific
    elif bvh_id == "CAPITALINTRODUCTION":
        score = 45 * safe_float(features.get("capital_intro_signal")) + 25 * network + 20 * safe_float(features.get("target_has_capital_terms")) + 10 * specific
    elif bvh_id == "INSIGHTEXCHANGE":
        score = 45 * insight + 25 * safe_float(features.get("insight_exchange_interaction")) + 20 * specific + 10 * pair_proto
    elif bvh_id == "RECRUITMENT":
        score = 60 * safe_float(features.get("recruitment_signal")) + 20 * safe_float(features.get("hiring_interaction")) + 20 * specific
    elif bvh_id == "TRUSTPATH":
        score = 45 * safe_float(features.get("trust_path_signal")) + 30 * trust + 15 * network + 10 * specific
    elif bvh_id == "ECOSYSTEMAMPLIFICATION":
        score = 35 * network_amp + 25 * network + 20 * safe_float(features.get("client_intro_signal")) + 20 * strategic
    else:
        score = 50 * safe_float(features.get("bvh_type_fit_score")) + 25 * specific + 25 * pair_proto

    return {
        "BVHSpecificEvidenceScore": round(max(0.0, min(100.0, score)), 2),
        "BVHSpecificEvidenceJSON": {
            "bvh_type_id": bvh_id,
            "product_need": round(product_need, 4),
            "network": round(network, 4),
            "insight": round(insight, 4),
            "strategic_alignment": round(strategic, 4),
            "capability_complementarity": round(complement, 4),
            "trust": round(trust, 4),
            "problem_fit": round(problem_fit, 4),
            "specific_overlap": round(specific, 4),
            "pair_prototype": round(pair_proto, 4),
            "supplier_match_signal": round(safe_float(features.get("supplier_match_signal")), 4),
            "client_intro_signal": round(safe_float(features.get("client_intro_signal")), 4),
            "capital_intro_signal": round(safe_float(features.get("capital_intro_signal")), 4),
            "partnership_signal": round(safe_float(features.get("partnership_signal")), 4),
            "trust_path_signal": round(safe_float(features.get("trust_path_signal")), 4),
            "recruitment_signal": round(safe_float(features.get("recruitment_signal")), 4),
        },
    }


def bvh_prototype_type_score(candidate: Dict[str, Any]) -> float:
    bvh_id = str(candidate.get("bvh_type", {}).get("BVHTypeID", "")).upper()
    proto_type = str(candidate.get("PairPrototypeMatchBVHTypeID", "")).upper()
    proto_score = safe_float(candidate.get("PairPrototypeScore"))
    if not proto_type:
        return 0.0
    if proto_type == bvh_id:
        return min(100.0, max(0.0, proto_score))
    if proto_score >= 65:
        return 5.0
    return 20.0


def bvh_confusion_penalty(candidate: Dict[str, Any], features: Dict[str, float], evidence_score: float) -> Dict[str, Any]:
    bvh_id = str(candidate.get("bvh_type", {}).get("BVHTypeID", "")).upper()
    proto_type = str(candidate.get("PairPrototypeMatchBVHTypeID", "")).upper()
    proto_score = safe_float(candidate.get("PairPrototypeScore"))
    penalty = 0.0
    reasons: List[str] = []

    if proto_type and proto_type != bvh_id and proto_score >= 70:
        penalty += 35.0
        reasons.append(f"Closest pair prototype says {proto_type}, not {bvh_id}.")

    if bvh_id in CONFUSION_CLUSTER_PRODUCT and evidence_score < 45:
        penalty += 12.0
        reasons.append(f"{bvh_id} is in product/problem/supplier/partnership confusion cluster with weak evidence.")

    if bvh_id == "PARTNERSHIP" and safe_float(features.get("product_need_interaction")) >= 1.0 and safe_float(features.get("partnership_signal")) < 0.25:
        penalty += 18.0
        reasons.append("PARTNERSHIP lacks reciprocal partnership signal; product/need looks more transactional.")

    if bvh_id == "SUPPLIERMATCH" and safe_float(features.get("supplier_match_signal")) < 0.25:
        penalty += 16.0
        reasons.append("SUPPLIERMATCH lacks supplier/procurement/vendor signal.")

    if bvh_id == "PRODUCTSALE" and safe_float(features.get("target_has_buying_terms")) < 0.5 and safe_float(features.get("product_need_interaction")) < 1.0:
        penalty += 16.0
        reasons.append("PRODUCTSALE lacks buyer/product-need evidence.")

    if bvh_id == "PROBLEMSOLUTION" and safe_float(features.get("problem_solution_fit")) < 0.50:
        penalty += 10.0
        reasons.append("PROBLEMSOLUTION has weak problem-solution fit.")

    if bvh_id in NETWORK_BVHS and safe_float(features.get("source_has_network_access")) < 1.0:
        penalty += 14.0
        reasons.append(f"{bvh_id} lacks strong network/access evidence.")

    return {
        "BVHConfusionPenalty": round(min(70.0, penalty), 2),
        "BVHConfusionPenaltyReason": " | ".join(reasons) if reasons else "No BVH confusion penalty.",
    }


def apply_bvh_precision_scores(candidate: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    base_features = candidate_features(candidate)
    features = enhance_bvh_features(base_features, candidate)
    bvh_id = str(candidate.get("bvh_type", {}).get("BVHTypeID", "")).upper()

    classifier = (model or {}).get("models", {}).get(bvh_id, {})
    prob = predict_label_probability({k: features.get(k, 0.0) for k in classifier.get("feature_names", features.keys())}, classifier) if classifier.get("enabled") else 0.0
    classifier_score = prob * 100.0

    evidence = bvh_specific_evidence_score(candidate, features)
    proto_type_score = bvh_prototype_type_score(candidate)
    confusion = bvh_confusion_penalty(candidate, features, safe_float(evidence["BVHSpecificEvidenceScore"]))

    canonical_raw = (
        0.35 * classifier_score
        + 0.25 * safe_float(evidence["BVHSpecificEvidenceScore"])
        + 0.20 * proto_type_score
        + 0.10 * safe_float(candidate.get("BVHTypeSupervisedScore"))
        + 0.10 * safe_float(candidate.get("HardBVHGateScore"), 100.0)
    )
    canonical_score = max(0.0, min(100.0, canonical_raw - safe_float(confusion["BVHConfusionPenalty"])))

    pair_final = (
        0.45 * safe_float(candidate.get("PairPrototypeScore"))
        + 0.30 * safe_float(candidate.get("SpecificPairOverlapScore"))
        + 0.25 * safe_float(candidate.get("PairSupervisedScore"))
        - safe_float(candidate.get("GenericPatternPenalty"))
    )
    pair_final = max(0.0, min(100.0, pair_final))

    final_before_saturation = 0.70 * pair_final + 0.30 * canonical_score

    trace = candidate.get("SupervisedFeatureTraceJSON", {}) or {}
    if not isinstance(trace, dict):
        trace = {}
    trace["bvh_precision1"] = {
        "bvh_type_id": bvh_id,
        "classifier_probability": round(prob, 6),
        "classifier_score": round(classifier_score, 2),
        "classifier_top_features": top_feature_contributions(features, classifier, limit=10) if classifier.get("enabled") else [],
        "specific_evidence": evidence,
        "bvh_prototype_type_score": round(proto_type_score, 2),
        "confusion_penalty": confusion,
        "pair_final_score": round(pair_final, 2),
        "formula": "BVHCanonicalScore = 0.35*BVHTypeClassifierScore + 0.25*BVHSpecificEvidenceScore + 0.20*BVHPrototypeTypeScore + 0.10*BVHTypeSupervisedScore + 0.10*HardBVHGateScore - BVHConfusionPenalty",
    }

    return {
        "BVHTypeClassifierProbability": round(prob, 6),
        "BVHTypeClassifierScore": round(classifier_score, 2),
        "BVHSpecificEvidenceScore": evidence["BVHSpecificEvidenceScore"],
        "BVHSpecificEvidenceJSON": evidence["BVHSpecificEvidenceJSON"],
        "BVHPrototypeTypeScore": round(proto_type_score, 2),
        "BVHConfusionPenalty": confusion["BVHConfusionPenalty"],
        "BVHConfusionPenaltyReason": confusion["BVHConfusionPenaltyReason"],
        "BVHCanonicalScoreRaw": round(canonical_raw, 2),
        "BVHCanonicalScore": round(canonical_score, 2),
        "PairFinalScore": round(pair_final, 2),
        "FinalRAINScoreBeforeSaturation": round(final_before_saturation, 2),
        "FinalRAINScore": round(final_before_saturation, 2),
        "SupervisedFeatureTraceJSON": trace,
    }


def annotate_bvh_margins(pair_candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_candidates = sorted(
        pair_candidates,
        key=lambda c: (
            safe_float(c.get("BVHCanonicalScore")),
            safe_float(c.get("BVHSpecificEvidenceScore")),
            safe_float(c.get("BVHTypeClassifierScore")),
            safe_float(c.get("FinalRAINScore")),
        ),
        reverse=True,
    )
    best = safe_float(sorted_candidates[0].get("BVHCanonicalScore")) if sorted_candidates else 0.0
    second = safe_float(sorted_candidates[1].get("BVHCanonicalScore")) if len(sorted_candidates) > 1 else 0.0
    margin = max(0.0, best - second)

    if margin >= 25:
        band = "HIGH"
    elif margin >= 12:
        band = "MEDIUM"
    elif margin >= 5:
        band = "LOW"
    else:
        band = "AMBIGUOUS"

    rank_by_id = {id(c): idx + 1 for idx, c in enumerate(sorted_candidates)}
    for c in pair_candidates:
        c["BestBVHTypeScore"] = round(best, 2)
        c["SecondBestBVHTypeScore"] = round(second, 2)
        c["BVHMarginScore"] = round(margin, 2)
        c["BVHConfidenceBand"] = band
        c["BVHCanonicalRankWithinPair"] = rank_by_id.get(id(c), 999)
    return sorted_candidates


def select_bvh_candidates_for_pair(
    pair_candidates: List[Dict[str, Any]],
    *,
    mode: str = "canonical_top1",
    top2_margin_threshold: float = 8.0,
    top2_min_canonical_score: float = 65.0,
    legacy_max_bvh_per_pair: int = 1,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if not pair_candidates:
        return [], {"input_count": 0, "selected_count": 0, "mode": mode}

    if mode == "legacy":
        ranked = sorted(pair_candidates, key=lambda c: (
            safe_float(c.get("FinalRAINScore")),
            safe_float(c.get("PairPrototypeScore")),
            safe_float(c.get("SpecificPairOverlapScore")),
        ), reverse=True)
        selected = ranked[:legacy_max_bvh_per_pair] if legacy_max_bvh_per_pair and legacy_max_bvh_per_pair > 0 else ranked
        for i, c in enumerate(ranked, 1):
            c["BVHCanonicalRankWithinPair"] = i
            c["BVHCanonicalKeep"] = c in selected
            c["BVHSelectionDecision"] = "LEGACY_KEEP" if c in selected else "LEGACY_DROP"
        return selected, {"input_count": len(pair_candidates), "selected_count": len(selected), "mode": mode}

    ranked = annotate_bvh_margins(pair_candidates)
    selected = [ranked[0]]

    if mode == "canonical_top2_margin" and len(ranked) > 1:
        margin = safe_float(ranked[0].get("BVHMarginScore"))
        second_score = safe_float(ranked[1].get("BVHCanonicalScore"))
        if margin <= top2_margin_threshold and second_score >= top2_min_canonical_score:
            selected.append(ranked[1])

    selected_ids = {id(c) for c in selected}
    for c in ranked:
        keep = id(c) in selected_ids
        c["BVHCanonicalKeep"] = keep
        if keep and safe_float(c.get("BVHCanonicalRankWithinPair")) == 1:
            c["BVHSelectionDecision"] = "KEEP_CANONICAL_TOP1"
        elif keep:
            c["BVHSelectionDecision"] = "KEEP_LOW_MARGIN_TOP2"
        else:
            c["BVHSelectionDecision"] = "DROP_NON_CANONICAL_BVH"

    return selected, {
        "input_count": len(pair_candidates),
        "selected_count": len(selected),
        "mode": mode,
        "best_bvh_type_id": ranked[0].get("bvh_type", {}).get("BVHTypeID", ""),
        "best_bvh_score": safe_float(ranked[0].get("BVHCanonicalScore")),
        "second_bvh_score": safe_float(ranked[1].get("BVHCanonicalScore")) if len(ranked) > 1 else 0.0,
        "margin": safe_float(ranked[0].get("BVHMarginScore")),
        "confidence_band": ranked[0].get("BVHConfidenceBand", ""),
    }


def bvh_precision_model_summary(model: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "enabled": model.get("enabled", False),
        "method": model.get("method", ""),
        "bvh_type_count": model.get("bvh_type_count", 0),
        "enabled_classifier_count": model.get("enabled_classifier_count", 0),
        "diagnostics": model.get("diagnostics", {}),
    }
