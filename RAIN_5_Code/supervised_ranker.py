from __future__ import annotations
from typing import Any, Dict, List, Tuple
import math
import random

from .calibration import pair_bvh_key, pair_key
from .utils import normalise_text, safe_float


# ---------------------------------------------------------------------------
# Shared maths
# ---------------------------------------------------------------------------

def sigmoid(x: float) -> float:
    if x >= 35:
        return 1.0
    if x <= -35:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def log_scale(value: float, denom: float = 7.0) -> float:
    value = max(1.0, float(value or 0.0))
    return min(1.0, max(0.0, math.log10(value) / denom))


def bool_feature(value: Any) -> float:
    return 1.0 if bool(value) else 0.0


# ---------------------------------------------------------------------------
# Positive label/value lookup
# ---------------------------------------------------------------------------

def get_positive_value_lookup(calibration: Dict[str, Any]) -> Dict[Tuple[Tuple[str, str], str], float]:
    values: Dict[Tuple[Tuple[str, str], str], float] = {}
    for ex in calibration.get("positive_examples", []) or []:
        key = ex.get("pair_bvh_key")
        if not key:
            key = pair_bvh_key(ex.get("PersonAName", ""), ex.get("PersonBName", ""), ex.get("BVHTypeID", ""))
        value = safe_float(ex.get("HVTI.ExpectedLifetimeValueGBP")) or safe_float(ex.get("HVTI.PotentialLifetimeValueGBP"))
        values[key] = max(values.get(key, 0.0), value)
    return values


def get_positive_pair_value_lookup(calibration: Dict[str, Any]) -> Dict[Tuple[str, str], float]:
    values: Dict[Tuple[str, str], float] = {}
    for ex in calibration.get("positive_examples", []) or []:
        key = ex.get("pair_key")
        if not key:
            key = pair_key(ex.get("PersonAName", ""), ex.get("PersonBName", ""))
        value = safe_float(ex.get("HVTI.ExpectedLifetimeValueGBP")) or safe_float(ex.get("HVTI.PotentialLifetimeValueGBP"))
        values[key] = max(values.get(key, 0.0), value)
    return values


def candidate_pair_key(candidate: Dict[str, Any]) -> Tuple[str, str]:
    source = candidate["source"]
    target = candidate["target"]
    return pair_key(source.get("full_name", ""), target.get("full_name", ""))


def candidate_pair_bvh_key(candidate: Dict[str, Any]) -> Tuple[Tuple[str, str], str]:
    source = candidate["source"]
    target = candidate["target"]
    bvh_type = candidate["bvh_type"]
    return pair_bvh_key(source.get("full_name", ""), target.get("full_name", ""), bvh_type.get("BVHTypeID", ""))


def candidate_pair_label(candidate: Dict[str, Any], calibration: Dict[str, Any]) -> int:
    return 1 if candidate_pair_key(candidate) in calibration.get("positive_pair_names", set()) else 0


def candidate_bvh_label(candidate: Dict[str, Any], calibration: Dict[str, Any]) -> int:
    return 1 if candidate_pair_bvh_key(candidate) in calibration.get("positive_pair_bvh_names", set()) else 0


# Backward-compatible name: exact pair+BVH label.
def candidate_label(candidate: Dict[str, Any], calibration: Dict[str, Any]) -> int:
    return candidate_bvh_label(candidate, calibration)


def candidate_positive_value(candidate: Dict[str, Any], calibration: Dict[str, Any]) -> float:
    return get_positive_value_lookup(calibration).get(candidate_pair_bvh_key(candidate), 0.0)


def candidate_pair_positive_value(candidate: Dict[str, Any], calibration: Dict[str, Any]) -> float:
    return get_positive_pair_value_lookup(calibration).get(candidate_pair_key(candidate), 0.0)


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def candidate_features(candidate: Dict[str, Any]) -> Dict[str, float]:
    params = candidate["hvti_result"].get("EstimatedHVTIParameters", {})
    evidence = candidate["selected_fit"].get("evidence", {})
    roi = candidate["roi_result"]
    components = candidate["component_scores"]
    bvh_id = str(candidate["bvh_type"].get("BVHTypeID", "")).upper()

    source = candidate.get("source", {})
    target = candidate.get("target", {})
    source_text = normalise_text(source.get("_search_text", "")) + " " + normalise_text(source.get("current_role", ""))
    target_text = normalise_text(target.get("_search_text", "")) + " " + normalise_text(target.get("current_role", ""))

    hiring_terms = ["hire", "hiring", "recruit", "recruitment", "talent", "candidate", "people", "hr", "workforce", "job"]
    buying_terms = ["buy", "buyer", "pain", "need", "procure", "vendor", "supplier", "solution", "cost", "risk"]
    network_terms = ["network", "introduce", "introduction", "access", "client", "relationship", "connector"]

    features: Dict[str, float] = {
        "bias": 1.0,

        # Current RAIN signals.
        "estimated_hvti_score": safe_float(candidate["hvti_result"].get("EstimatedHVTIScore")) / 100.0,
        "pair_match_score": safe_float(components.get("PairMatchScore")) / 100.0,
        "bvh_type_fit_score": safe_float(components.get("BVHTypeFitScore")) / 100.0,
        "value_score": safe_float(components.get("ValueScore")) / 100.0,
        "ranking_score": safe_float(candidate.get("RankingScore")) / 100.0,

        # Value signals.
        "estimated_potential_usd_log": log_scale(safe_float(roi.get("EstimatedHVITUSDValue"))),
        "estimated_expected_usd_log": log_scale(safe_float(roi.get("EstimatedExpectedUSDValue"))),
        "estimated_probability": safe_float(candidate["hvti_result"].get("EstimatedProbabilityOfTransaction")),

        # Core HVTI parameters.
        "problem_solution_fit": safe_float(params.get("problem_solution_fit")) / 100.0,
        "commercial_opportunity_size": safe_float(params.get("commercial_opportunity_size")) / 100.0,
        "urgency": safe_float(params.get("urgency")) / 100.0,
        "decision_authority": safe_float(params.get("decision_authority")) / 100.0,
        "capability_complementarity": safe_float(params.get("capability_complementarity")) / 100.0,
        "trust_potential": safe_float(params.get("trust_potential")) / 100.0,
        "strategic_alignment": safe_float(params.get("strategic_alignment")) / 100.0,
        "network_amplification": safe_float(params.get("network_amplification")) / 100.0,
        "timing_context": safe_float(params.get("timing_context")) / 100.0,
        "execution_probability": safe_float(params.get("execution_probability")) / 100.0,

        # Evidence booleans.
        "semantic_overlap": safe_float(evidence.get("semantic_overlap")),
        "same_primary_driver": bool_feature(evidence.get("same_primary_driver")),
        "same_market_regime": bool_feature(evidence.get("same_market_regime")),
        "same_location": bool_feature(evidence.get("same_location")),
        "same_organisation": bool_feature(evidence.get("same_organisation")),
        "source_has_product_or_solution": bool_feature(evidence.get("source_has_product_or_solution")),
        "target_has_need_or_pain": bool_feature(evidence.get("target_has_need_or_pain")),
        "source_has_insight": bool_feature(evidence.get("source_has_insight")),
        "source_has_network_access": bool_feature(evidence.get("source_has_network_access")),
        "explicit_value_signal_gbp_log": log_scale(safe_float(evidence.get("explicit_value_signal_gbp")), denom=7.0),

        # Textual guardrail hints.
        "source_has_hiring_terms": bool_feature(any(t in source_text for t in hiring_terms)),
        "target_has_hiring_terms": bool_feature(any(t in target_text for t in hiring_terms)),
        "source_has_buying_terms": bool_feature(any(t in source_text for t in buying_terms)),
        "target_has_buying_terms": bool_feature(any(t in target_text for t in buying_terms)),
        "source_has_network_terms": bool_feature(any(t in source_text for t in network_terms)),
        "target_has_network_terms": bool_feature(any(t in target_text for t in network_terms)),
    }

    for known in [
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
    ]:
        features[f"bvh_{known}"] = 1.0 if bvh_id == known else 0.0

    features["product_need_interaction"] = (
        features["source_has_product_or_solution"] * features["target_has_need_or_pain"]
    )
    features["network_intro_interaction"] = (
        features["source_has_network_access"]
        * max(features["bvh_CLIENTINTRODUCTION"], features["bvh_CAPITALINTRODUCTION"], features["bvh_ECOSYSTEMAMPLIFICATION"])
    )
    features["insight_exchange_interaction"] = features["source_has_insight"] * features["bvh_INSIGHTEXCHANGE"]
    features["hiring_interaction"] = max(features["source_has_hiring_terms"], features["target_has_hiring_terms"]) * features["bvh_RECRUITMENT"]
    features["pair_core_fit"] = (
        features["pair_match_score"]
        + features["problem_solution_fit"]
        + features["capability_complementarity"]
        + features["strategic_alignment"]
    ) / 4.0
    features["bvh_core_fit"] = (
        features["bvh_type_fit_score"]
        + features["product_need_interaction"]
        + features["network_intro_interaction"]
        + features["insight_exchange_interaction"]
        + features["hiring_interaction"]
    ) / 5.0

    return features


def pair_feature_subset(features: Dict[str, float]) -> Dict[str, float]:
    """
    Pair model should not overfit to a specific BVHType. Keep pair-level and general commercial features.
    """
    excluded_prefixes = ("bvh_",)
    excluded_names = {
        "bvh_type_fit_score",
        "bvh_core_fit",
        "network_intro_interaction",
        "insight_exchange_interaction",
        "hiring_interaction",
    }
    return {
        k: v for k, v in features.items()
        if not k.startswith(excluded_prefixes) and k not in excluded_names
    }


def bvh_feature_subset(features: Dict[str, float]) -> Dict[str, float]:
    return dict(features)


def candidate_training_row(candidate: Dict[str, Any], calibration: Dict[str, Any]) -> Dict[str, Any]:
    source = candidate["source"]
    target = candidate["target"]
    bvh_type = candidate["bvh_type"]
    pair_label = candidate_pair_label(candidate, calibration)
    bvh_label = candidate_bvh_label(candidate, calibration)
    features = candidate_features(candidate)

    return {
        "PersonAName": source.get("full_name", ""),
        "PersonBName": target.get("full_name", ""),
        "BVHTypeID": bvh_type.get("BVHTypeID", ""),
        "Label": bvh_label,
        "PairLabel": pair_label,
        "BVHTypeLabel": bvh_label,
        "PositiveExpectedValueGBP": candidate_positive_value(candidate, calibration),
        "PairPositiveExpectedValueGBP": candidate_pair_positive_value(candidate, calibration),
        "features": features,
        "pair_features": pair_feature_subset(features),
        "bvh_features": bvh_feature_subset(features),
    }


def make_pair_training_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["Label"] = int(r.get("PairLabel", 0))
        rr["PositiveExpectedValueGBP"] = safe_float(r.get("PairPositiveExpectedValueGBP"))
        rr["features"] = dict(r.get("pair_features", r.get("features", {})))
        out.append(rr)
    return out


def make_bvh_training_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        rr = dict(r)
        rr["Label"] = int(r.get("BVHTypeLabel", r.get("Label", 0)))
        rr["PositiveExpectedValueGBP"] = safe_float(r.get("PositiveExpectedValueGBP"))
        rr["features"] = dict(r.get("bvh_features", r.get("features", {})))
        out.append(rr)
    return out


# ---------------------------------------------------------------------------
# Logistic training
# ---------------------------------------------------------------------------

def sample_training_rows(
    rows: List[Dict[str, Any]],
    negative_samples_per_positive: int,
    max_training_rows: int,
    random_seed: int,
) -> List[Dict[str, Any]]:
    positives = [r for r in rows if int(r.get("Label", 0)) == 1]
    negatives = [r for r in rows if int(r.get("Label", 0)) == 0]

    rng = random.Random(random_seed)
    rng.shuffle(negatives)

    target_negative_count = len(positives) * max(1, int(negative_samples_per_positive))
    if not positives:
        target_negative_count = min(len(negatives), max_training_rows)

    selected = positives + negatives[:target_negative_count]

    if max_training_rows and len(selected) > max_training_rows:
        keep_negatives = max(0, max_training_rows - len(positives))
        selected = positives + negatives[:keep_negatives]

    rng.shuffle(selected)
    return selected


def training_weight(row: Dict[str, Any], high_value_positive_weight: float) -> float:
    if int(row.get("Label", 0)) != 1:
        return 1.0
    value = safe_float(row.get("PositiveExpectedValueGBP"))
    value_multiplier = 1.0 + min(high_value_positive_weight, math.log10(max(10.0, value)) / 6.0)
    return value_multiplier


def train_logistic_ranker(
    training_rows_all: List[Dict[str, Any]],
    negative_samples_per_positive: int = 10,
    epochs: int = 80,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    max_training_rows: int = 5000,
    random_seed: int = 17,
    high_value_positive_weight: float = 2.5,
    model_name: str = "logistic_ranker",
) -> Dict[str, Any]:
    if not training_rows_all:
        return {
            "enabled": False,
            "reason": "No training rows available.",
            "model_name": model_name,
            "weights": {},
            "feature_names": [],
        }

    feature_names = sorted(training_rows_all[0]["features"].keys())
    training_rows = sample_training_rows(
        training_rows_all,
        negative_samples_per_positive=negative_samples_per_positive,
        max_training_rows=max_training_rows,
        random_seed=random_seed,
    )

    positive_count_all = sum(1 for r in training_rows_all if int(r.get("Label", 0)) == 1)
    negative_count_all = len(training_rows_all) - positive_count_all
    positive_count_train = sum(1 for r in training_rows if int(r.get("Label", 0)) == 1)
    negative_count_train = len(training_rows) - positive_count_train

    if positive_count_train == 0:
        return {
            "enabled": False,
            "reason": "No positive labels in sampled training data.",
            "model_name": model_name,
            "weights": {},
            "feature_names": feature_names,
            "training_row_count_all": len(training_rows_all),
            "positive_count_all": positive_count_all,
            "negative_count_all": negative_count_all,
        }

    weights = {name: 0.0 for name in feature_names}

    for _epoch in range(max(1, int(epochs))):
        for row in training_rows:
            y = float(row.get("Label", 0))
            x = row["features"]
            z = sum(weights[name] * float(x.get(name, 0.0)) for name in feature_names)
            p = sigmoid(z)
            err = y - p
            row_weight = training_weight(row, high_value_positive_weight)

            for name in feature_names:
                xv = float(x.get(name, 0.0))
                grad = row_weight * err * xv - l2 * weights[name]
                weights[name] += learning_rate * grad

    preds = []
    for row in training_rows:
        score = predict_label_probability(row["features"], {"enabled": True, "weights": weights, "feature_names": feature_names})
        preds.append((score, int(row.get("Label", 0))))

    positives = [p for p, y in preds if y == 1]
    negatives = [p for p, y in preds if y == 0]

    return {
        "enabled": True,
        "model_name": model_name,
        "method": "pure_python_weighted_logistic_regression",
        "feature_names": feature_names,
        "weights": {k: round(v, 6) for k, v in weights.items()},
        "training_row_count_all": len(training_rows_all),
        "training_row_count_used": len(training_rows),
        "positive_count_all": positive_count_all,
        "negative_count_all": negative_count_all,
        "positive_count_train": positive_count_train,
        "negative_count_train": negative_count_train,
        "negative_samples_per_positive": negative_samples_per_positive,
        "epochs": epochs,
        "learning_rate": learning_rate,
        "l2": l2,
        "max_training_rows": max_training_rows,
        "random_seed": random_seed,
        "high_value_positive_weight": high_value_positive_weight,
        "diagnostics": {
            "avg_positive_score_train": round(sum(positives) / len(positives), 6) if positives else 0.0,
            "avg_negative_score_train": round(sum(negatives) / len(negatives), 6) if negatives else 0.0,
            "min_positive_score_train": round(min(positives), 6) if positives else 0.0,
            "max_negative_score_train": round(max(negatives), 6) if negatives else 0.0,
        },
    }


def predict_label_probability(features: Dict[str, float], model: Dict[str, Any]) -> float:
    if not model or not model.get("enabled"):
        return 0.0
    weights = model.get("weights", {})
    feature_names = model.get("feature_names", sorted(features.keys()))
    z = sum(float(weights.get(name, 0.0)) * float(features.get(name, 0.0)) for name in feature_names)
    return sigmoid(z)


def top_feature_contributions(features: Dict[str, float], model: Dict[str, Any], limit: int = 8) -> List[Dict[str, Any]]:
    if not model or not model.get("enabled"):
        return []
    weights = model.get("weights", {})
    rows = []
    for name, value in features.items():
        contribution = float(weights.get(name, 0.0)) * float(value)
        if abs(contribution) > 0.0001:
            rows.append({
                "feature": name,
                "value": round(float(value), 6),
                "weight": round(float(weights.get(name, 0.0)), 6),
                "contribution": round(contribution, 6),
            })
    rows.sort(key=lambda r: abs(r["contribution"]), reverse=True)
    return rows[:limit]


# ---------------------------------------------------------------------------
# Hard BVH false-positive gates
# ---------------------------------------------------------------------------

def hard_bvh_gate(candidate: Dict[str, Any], features: Dict[str, float]) -> Dict[str, Any]:
    bvh_id = str(candidate["bvh_type"].get("BVHTypeID", "")).upper()

    source_product = features.get("source_has_product_or_solution", 0.0) >= 1.0
    target_need = features.get("target_has_need_or_pain", 0.0) >= 1.0
    source_network = features.get("source_has_network_access", 0.0) >= 1.0 or features.get("source_has_network_terms", 0.0) >= 1.0
    source_insight = features.get("source_has_insight", 0.0) >= 1.0
    same_driver = features.get("same_primary_driver", 0.0) >= 1.0
    same_regime = features.get("same_market_regime", 0.0) >= 1.0
    hiring = features.get("source_has_hiring_terms", 0.0) >= 1.0 or features.get("target_has_hiring_terms", 0.0) >= 1.0

    passed = True
    multiplier = 1.0
    reason = "PASS"

    if bvh_id == "PRODUCTSALE":
        if not (source_product and target_need):
            passed, multiplier, reason = False, 0.20, "PRODUCTSALE requires source product/solution and target need/pain."
    elif bvh_id == "PROBLEMSOLUTION":
        if not ((source_product or source_insight) and target_need):
            passed, multiplier, reason = False, 0.30, "PROBLEMSOLUTION requires source product/insight and target need/pain."
    elif bvh_id in {"CLIENTINTRODUCTION", "CAPITALINTRODUCTION", "ECOSYSTEMAMPLIFICATION"}:
        if not source_network:
            passed, multiplier, reason = False, 0.25, f"{bvh_id} requires source network/access signal."
        elif not (target_need or same_driver or same_regime):
            passed, multiplier, reason = False, 0.45, f"{bvh_id} requires target need or strong contextual alignment."
    elif bvh_id == "INSIGHTEXCHANGE":
        if not source_insight:
            passed, multiplier, reason = False, 0.35, "INSIGHTEXCHANGE requires source insight signal."
    elif bvh_id == "RECRUITMENT":
        if not hiring:
            passed, multiplier, reason = False, 0.15, "RECRUITMENT requires explicit hiring/recruitment/talent signal."
    elif bvh_id == "SUPPLIERMATCH":
        if not (source_product and target_need):
            passed, multiplier, reason = False, 0.25, "SUPPLIERMATCH requires source supply/product and target need/pain."
    elif bvh_id == "PARTNERSHIP":
        if not (same_driver or same_regime or features.get("strategic_alignment", 0.0) >= 0.65):
            passed, multiplier, reason = False, 0.50, "PARTNERSHIP requires strategic context alignment."
    elif bvh_id == "TRUSTPATH":
        if not (same_driver or features.get("trust_potential", 0.0) >= 0.60 or features.get("same_organisation", 0.0) >= 1.0):
            passed, multiplier, reason = False, 0.50, "TRUSTPATH requires trust/context signal."

    return {
        "HardBVHGatePass": passed,
        "HardBVHGateMultiplier": round(multiplier, 4),
        "HardBVHGateScore": round(multiplier * 100.0, 2),
        "HardBVHGateReason": reason,
    }


# ---------------------------------------------------------------------------
# Precision@50 pair-first scoring
# ---------------------------------------------------------------------------

def train_precision50_models(
    training_rows_all: List[Dict[str, Any]],
    negative_samples_per_positive: int = 10,
    epochs: int = 80,
    learning_rate: float = 0.08,
    l2: float = 0.001,
    max_training_rows: int = 5000,
    random_seed: int = 17,
    high_value_positive_weight: float = 2.5,
) -> Dict[str, Any]:
    pair_rows = make_pair_training_rows(training_rows_all)
    bvh_rows = make_bvh_training_rows(training_rows_all)

    pair_model = train_logistic_ranker(
        pair_rows,
        negative_samples_per_positive=negative_samples_per_positive,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        max_training_rows=max_training_rows,
        random_seed=random_seed,
        high_value_positive_weight=high_value_positive_weight,
        model_name="pair_supervised_model",
    )
    bvh_model = train_logistic_ranker(
        bvh_rows,
        negative_samples_per_positive=negative_samples_per_positive,
        epochs=epochs,
        learning_rate=learning_rate,
        l2=l2,
        max_training_rows=max_training_rows,
        random_seed=random_seed + 101,
        high_value_positive_weight=high_value_positive_weight,
        model_name="bvh_type_supervised_model",
    )

    return {
        "enabled": bool(pair_model.get("enabled") and bvh_model.get("enabled")),
        "method": "Precision@50 pair-first dual supervised logistic ranker",
        "pair_model": pair_model,
        "bvh_type_model": bvh_model,
        "training_row_count_all": len(training_rows_all),
        "pair_positive_count_all": sum(1 for r in training_rows_all if int(r.get("PairLabel", 0)) == 1),
        "bvh_positive_count_all": sum(1 for r in training_rows_all if int(r.get("BVHTypeLabel", r.get("Label", 0))) == 1),
        "formula": "FinalRAINScore = (0.50*PairSupervisedScore + 0.30*BVHTypeSupervisedScore + 0.10*ValueScore + 0.10*RankingScore) * HardBVHGateMultiplier",
    }


def apply_precision50_scores(candidate: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    features = candidate_features(candidate)
    pair_features = pair_feature_subset(features)
    bvh_features = bvh_feature_subset(features)

    pair_model = model.get("pair_model", {}) if model else {}
    bvh_model = model.get("bvh_type_model", {}) if model else {}

    pair_probability = predict_label_probability(pair_features, pair_model)
    bvh_probability = predict_label_probability(bvh_features, bvh_model)
    pair_score = round(pair_probability * 100.0, 2)
    bvh_score = round(bvh_probability * 100.0, 2)

    gate = hard_bvh_gate(candidate, features)

    ranking_score = safe_float(candidate.get("RankingScore"))
    value_score = safe_float(candidate["component_scores"].get("ValueScore"))

    raw_final = (
        0.50 * pair_score
        + 0.30 * bvh_score
        + 0.10 * value_score
        + 0.10 * ranking_score
    )
    final_score = round(raw_final * safe_float(gate.get("HardBVHGateMultiplier"), 1.0), 2)

    combined_probability = (0.625 * pair_probability) + (0.375 * bvh_probability)

    return {
        "PairSupervisedProbability": round(pair_probability, 6),
        "PairSupervisedScore": pair_score,
        "BVHTypeSupervisedProbability": round(bvh_probability, 6),
        "BVHTypeSupervisedScore": bvh_score,

        # Backward-compatible V2 fields.
        "SupervisedLabelProbability": round(combined_probability, 6),
        "SupervisedLabelScore": round(combined_probability * 100.0, 2),

        "FinalRAINScoreRaw": round(raw_final, 2),
        "FinalRAINScore": final_score,
        **gate,
        "SupervisedFeatureTraceJSON": {
            "features": {k: round(v, 6) for k, v in features.items()},
            "pair_top_contributions": top_feature_contributions(pair_features, pair_model),
            "bvh_top_contributions": top_feature_contributions(bvh_features, bvh_model),
            "hard_bvh_gate": gate,
        },
    }


# Backward-compatible wrapper used by earlier patch code.
def apply_supervised_scores(candidate: Dict[str, Any], model: Dict[str, Any]) -> Dict[str, Any]:
    if model and "pair_model" in model and "bvh_type_model" in model:
        return apply_precision50_scores(candidate, model)

    features = candidate_features(candidate)
    probability = predict_label_probability(features, model)
    supervised_label_score = round(probability * 100.0, 2)
    existing_ranking_score = safe_float(candidate.get("RankingScore"))
    value_score = safe_float(candidate["component_scores"].get("ValueScore"))
    final_score = round(
        0.60 * supervised_label_score
        + 0.20 * existing_ranking_score
        + 0.20 * value_score,
        2,
    )

    return {
        "PairSupervisedProbability": 0.0,
        "PairSupervisedScore": 0.0,
        "BVHTypeSupervisedProbability": round(probability, 6),
        "BVHTypeSupervisedScore": supervised_label_score,
        "SupervisedLabelProbability": round(probability, 6),
        "SupervisedLabelScore": supervised_label_score,
        "FinalRAINScoreRaw": final_score,
        "FinalRAINScore": final_score,
        "HardBVHGatePass": True,
        "HardBVHGateMultiplier": 1.0,
        "HardBVHGateScore": 100.0,
        "HardBVHGateReason": "Legacy supervised scoring path.",
        "SupervisedFeatureTraceJSON": {
            "features": {k: round(v, 6) for k, v in features.items()},
            "top_contributions": top_feature_contributions(features, model),
        },
    }
