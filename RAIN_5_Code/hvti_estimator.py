from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
from .utils import safe_float, normalise_text

SENIORITY_SCORES = {
    "Analyst": 35,
    "Associate": 45,
    "Vice President": 60,
    "Director": 72,
    "Managing Director": 86,
    "Partner": 88,
    "Founder": 84,
    "C-suite": 95,
}

HIGH_AUTHORITY_TERMS = ["head", "chief", "cfo", "coo", "cto", "ciso", "partner", "director", "founder", "managing director", "lead"]

def overlap_score(a_tokens: set, b_tokens: set) -> float:
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / max(1, len(a_tokens | b_tokens))

def has_any(profile: Dict[str, Any], fields: List[str]) -> bool:
    for field in fields:
        value = profile.get(field)
        if isinstance(value, list) and value:
            return True
        if isinstance(value, dict) and value:
            return True
        if isinstance(value, str) and value.strip():
            return True
    return False

def product_price(profile: Dict[str, Any]) -> float:
    pricing = profile.get("commercial_pricing_json") or {}
    products = pricing.get("products", []) if isinstance(pricing, dict) else []
    best = 0.0
    for product in products:
        if not isinstance(product, dict):
            continue
        best = max(
            best,
            safe_float(product.get("enterprise_price_gbp")),
            safe_float(product.get("target_price_gbp")),
            safe_float(product.get("min_price_gbp")),
            safe_float(product.get("enterprise_annual_price_gbp")),
            safe_float(product.get("typical_annual_price_gbp")),
        )
    return best

def product_value_case(profile: Dict[str, Any]) -> float:
    cases = profile.get("product_value_cases_json") or []
    best = 0.0
    for case in cases if isinstance(cases, list) else []:
        if isinstance(case, dict):
            best = max(best, safe_float(case.get("estimated_annual_value_gbp")))
    return best

def buyer_pain_cost(profile: Dict[str, Any]) -> float:
    pains = profile.get("buyer_pain_points_json") or []
    best = 0.0
    for pain in pains if isinstance(pains, list) else []:
        if isinstance(pain, dict):
            best = max(best, safe_float(pain.get("estimated_cost_of_inaction_gbp")))
    return best

def seniority_score(profile: Dict[str, Any]) -> float:
    seniority = profile.get("seniority", "")
    base = SENIORITY_SCORES.get(seniority, 55)
    role = normalise_text(profile.get("current_role", ""))
    if any(term in role for term in HIGH_AUTHORITY_TERMS):
        base = max(base, 78)
    return float(min(100, base))

def estimate_directional_fit(person_a: Dict[str, Any], person_b: Dict[str, Any], bvh_type: Dict[str, Any]) -> Dict[str, Any]:
    """
    Estimate A->B directional profile signals for this BVHType.
    For symmetric types the caller should also score B->A.
    """
    bvh_id = bvh_type["BVHTypeID"]
    a_tokens = person_a.get("_tokens", set())
    b_tokens = person_b.get("_tokens", set())
    semantic_overlap = overlap_score(a_tokens, b_tokens)

    same_driver = person_a.get("primary_driver_id") == person_b.get("primary_driver_id")
    same_market_regime = person_a.get("market_regime") == person_b.get("market_regime")
    same_location = person_a.get("location") == person_b.get("location")
    same_org = person_a.get("organisation_name") == person_b.get("organisation_name")

    a_has_product = product_price(person_a) > 0 or has_any(person_a, ["solutions_json", "commercial_pricing_json"])
    b_has_need = has_any(person_b, ["needs_json", "buyer_pain_points_json", "use_cases_json"])
    a_has_insight = has_any(person_a, ["insights_json", "credibility_statements_json"])
    a_has_network = "network" in normalise_text(person_a.get("_search_text", "")) or "capital_connector" in normalise_text(person_a.get("persona_cluster", ""))
    explicit_value = max(product_price(person_a), product_value_case(person_a), buyer_pain_cost(person_b))

    problem_solution_fit = 35 + semantic_overlap * 45 + (15 if same_driver else 0) + (10 if a_has_product and b_has_need else 0)
    commercial_size = 35 + min(45, explicit_value / 25000) + (10 if product_price(person_a) > 0 else 0)
    urgency = 35 + (18 if has_any(person_b, ["regulatory_drivers_json", "buyer_pain_points_json"]) else 0) + (10 if same_market_regime else 0)
    decision_authority = (seniority_score(person_a) + seniority_score(person_b)) / 2.0
    complementarity = 40 + (20 if a_has_product and b_has_need else 0) + (15 if a_has_insight and b_has_need else 0) + semantic_overlap * 20
    trust = 35 + (20 if same_org else 0) + (10 if same_driver else 0) + (10 if same_location else 0)
    strategic = 35 + (20 if same_driver else 0) + (12 if same_market_regime else 0) + semantic_overlap * 25
    network = 35 + (25 if a_has_network else 0) + (10 if "network" in normalise_text(person_b.get("_search_text", "")) else 0)
    timing = 45 + (15 if same_market_regime else 0) + (10 if same_location else 0)
    execution = 35 + (15 if product_price(person_a) > 0 else 0) + (15 if has_any(person_a, ["credibility_statements_json"]) else 0) + (10 if b_has_need else 0)

    # BVHType-specific boosts and penalties. These make BVHTypeFit less generic.
    if bvh_id == "PRODUCTSALE":
        problem_solution_fit += 18 if a_has_product and b_has_need else -12
        commercial_size += 12
        execution += 8 if product_price(person_a) > 0 else -5
    elif bvh_id == "CLIENTINTRODUCTION":
        network += 25 if a_has_network else -8
        execution += 8
    elif bvh_id == "PROBLEMSOLUTION":
        problem_solution_fit += 18 if a_has_product and b_has_need else -8
        urgency += 10
    elif bvh_id == "INSIGHTEXCHANGE":
        problem_solution_fit += 12 if a_has_insight else -6
        commercial_size -= 8
        trust += 5
    elif bvh_id == "CAPITALINTRODUCTION":
        network += 18 if a_has_network else -5
        commercial_size += 8
    elif bvh_id == "PARTNERSHIP":
        complementarity += 18
        strategic += 12
    elif bvh_id == "RECRUITMENT":
        decision_authority += 5
        commercial_size = max(commercial_size, 45)
    elif bvh_id == "SUPPLIERMATCH":
        problem_solution_fit += 10 if a_has_product else -5
        execution += 12
    elif bvh_id == "TRUSTPATH":
        trust += 25
        execution += 5
    elif bvh_id == "ECOSYSTEMAMPLIFICATION":
        network += 25
        strategic += 10

    params = {
        "problem_solution_fit": min(100, max(0, problem_solution_fit)),
        "commercial_opportunity_size": min(100, max(0, commercial_size)),
        "urgency": min(100, max(0, urgency)),
        "decision_authority": min(100, max(0, decision_authority)),
        "capability_complementarity": min(100, max(0, complementarity)),
        "trust_potential": min(100, max(0, trust)),
        "strategic_alignment": min(100, max(0, strategic)),
        "network_amplification": min(100, max(0, network)),
        "timing_context": min(100, max(0, timing)),
        "execution_probability": min(100, max(0, execution)),
    }

    evidence = {
        "semantic_overlap": round(semantic_overlap, 4),
        "same_primary_driver": same_driver,
        "same_market_regime": same_market_regime,
        "same_location": same_location,
        "same_organisation": same_org,
        "source_has_product_or_solution": a_has_product,
        "target_has_need_or_pain": b_has_need,
        "source_has_insight": a_has_insight,
        "source_has_network_access": a_has_network,
        "explicit_value_signal_gbp": explicit_value,
    }

    component_preview = estimate_component_scores(params, evidence, bvh_id, estimated_usd_value=0)

    return {
        "direction": "A_TO_B",
        "parameters": {k: round(v, 2) for k, v in params.items()},
        "evidence": evidence,
        "component_preview": component_preview,
    }

def score_hvti(parameters: Dict[str, float], hvti_structure: Dict[str, Any], calibration: Dict[str, Any], bvh_type_id: str) -> Dict[str, Any]:
    weights = {p["ParameterID"]: float(p.get("DefaultWeight", 0)) for p in hvti_structure["parameters"]}
    total_weight = sum(weights.values()) or 1.0
    weighted = sum(float(parameters.get(pid, 0)) * w for pid, w in weights.items()) / total_weight

    prior = calibration.get("bvh_type_priors", {}).get(bvh_type_id, 0.0)
    historical_score = calibration.get("bvh_type_avg_score", {}).get(bvh_type_id)
    if historical_score:
        weighted = (weighted * 0.88) + (float(historical_score) * 0.07) + (prior * 100.0 * 0.05)
    else:
        weighted = (weighted * 0.95) + (prior * 100.0 * 0.05)

    probability_cfg = hvti_structure.get("probability_model", {})
    min_p = float(probability_cfg.get("min_probability", 0.05))
    max_p = float(probability_cfg.get("max_probability", 0.65))
    probability = min_p + (weighted / 100.0) * (max_p - min_p)

    return {
        "EstimatedHVTIParameters": {k: round(v, 2) for k, v in parameters.items()},
        "EstimatedHVTIScore": round(weighted, 2),
        "EstimatedProbabilityOfTransaction": round(probability, 4),
        "Calibration": {
            "label_count": calibration.get("label_count", 0),
            "bvh_type_prior": round(prior, 4),
            "historical_avg_score_used": historical_score is not None,
        }
    }

def estimate_component_scores(
    parameters: Dict[str, float],
    evidence: Dict[str, Any],
    bvh_type_id: str,
    estimated_usd_value: float,
) -> Dict[str, float]:
    """
    Decompose the RAIN5 score into business-facing components.

    PairMatchScore asks: should these two people meet?
    BVHTypeFitScore asks: is this the right reason / BVHType?
    ValueScore asks: is the meeting economically worth the slot?
    """
    pair_fields = [
        "problem_solution_fit",
        "decision_authority",
        "capability_complementarity",
        "trust_potential",
        "strategic_alignment",
        "timing_context",
        "execution_probability",
    ]
    pair_score = sum(float(parameters.get(k, 0)) for k in pair_fields) / len(pair_fields)

    base_bvh_fields = {
        "PRODUCTSALE": ["problem_solution_fit", "commercial_opportunity_size", "decision_authority", "execution_probability"],
        "CLIENTINTRODUCTION": ["network_amplification", "trust_potential", "decision_authority", "execution_probability"],
        "PROBLEMSOLUTION": ["problem_solution_fit", "urgency", "capability_complementarity", "execution_probability"],
        "INSIGHTEXCHANGE": ["problem_solution_fit", "trust_potential", "strategic_alignment", "decision_authority"],
        "CAPITALINTRODUCTION": ["network_amplification", "commercial_opportunity_size", "decision_authority", "execution_probability"],
        "PARTNERSHIP": ["capability_complementarity", "strategic_alignment", "trust_potential", "execution_probability"],
        "RECRUITMENT": ["decision_authority", "capability_complementarity", "execution_probability", "trust_potential"],
        "SUPPLIERMATCH": ["problem_solution_fit", "commercial_opportunity_size", "execution_probability", "decision_authority"],
        "TRUSTPATH": ["trust_potential", "strategic_alignment", "timing_context", "execution_probability"],
        "ECOSYSTEMAMPLIFICATION": ["network_amplification", "strategic_alignment", "commercial_opportunity_size", "execution_probability"],
    }
    fields = base_bvh_fields.get(bvh_type_id, ["problem_solution_fit", "commercial_opportunity_size", "execution_probability"])
    bvh_score = sum(float(parameters.get(k, 0)) for k in fields) / len(fields)

    # Evidence bonuses make the score reflect structured proof, not just text overlap.
    if evidence.get("source_has_product_or_solution") and evidence.get("target_has_need_or_pain"):
        bvh_score += 8
    if evidence.get("source_has_network_access") and bvh_type_id in ["CLIENTINTRODUCTION", "CAPITALINTRODUCTION", "ECOSYSTEMAMPLIFICATION"]:
        bvh_score += 8
    if evidence.get("same_primary_driver"):
        pair_score += 4
        bvh_score += 3
    if evidence.get("same_organisation"):
        pair_score += 3

    # Log scale so £10m does not dominate £1m by 10x, but still outranks low-value ideas.
    if estimated_usd_value and estimated_usd_value > 0:
        value_score = min(100.0, max(0.0, math.log10(max(1.0, estimated_usd_value)) / 7.0 * 100.0))
    else:
        value_score = min(100.0, max(0.0, float(parameters.get("commercial_opportunity_size", 0))))

    return {
        "PairMatchScore": round(min(100, max(0, pair_score)), 2),
        "BVHTypeFitScore": round(min(100, max(0, bvh_score)), 2),
        "ValueScore": round(min(100, max(0, value_score)), 2),
    }

def final_ranking_score(hvti_score: float, component_scores: Dict[str, float]) -> float:
    return round(
        0.45 * float(hvti_score)
        + 0.35 * float(component_scores.get("ValueScore", 0))
        + 0.20 * float(component_scores.get("BVHTypeFitScore", 0)),
        2,
    )
