from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Tuple
from .utils import read_csv, safe_float, normalise_text

def pair_key(name_a: Any, name_b: Any) -> Tuple[str, str]:
    return tuple(sorted([normalise_text(name_a), normalise_text(name_b)]))

def pair_bvh_key(name_a: Any, name_b: Any, bvh_type_id: Any) -> Tuple[Tuple[str, str], str]:
    return (pair_key(name_a, name_b), normalise_text(bvh_type_id).replace(" ", "").upper())

def load_labelled_matches(matches_created_path: Path | None) -> Dict[str, Any]:
    """
    Load matchesCreated as labelled positive examples.

    RAIN5.0 uses this for:
    - BVHType priors
    - historical score/value anchors
    - supervised HVTI weight tuning
    - positive pair/BVH lookup for backtests and training
    """
    if not matches_created_path or not matches_created_path.exists():
        return {
            "label_count": 0,
            "bvh_type_priors": {},
            "bvh_type_avg_score": {},
            "bvh_type_avg_value_gbp": {},
            "positive_pair_names": set(),
            "positive_pair_bvh_names": set(),
            "positive_examples": [],
        }

    rows = read_csv(matches_created_path)
    by_type: Dict[str, List[Dict[str, Any]]] = {}
    positive_pair_names = set()
    positive_pair_bvh_names = set()
    positive_examples: List[Dict[str, Any]] = []

    for r in rows:
        person_a = r.get("PersonAName", "")
        person_b = r.get("PersonBName", "")
        bvh_id = r.get("BVHCT.ID", "") or r.get("BVHTypeID", "")
        by_type.setdefault(bvh_id, []).append(r)

        if person_a and person_b:
            pk = pair_key(person_a, person_b)
            pbk = pair_bvh_key(person_a, person_b, bvh_id)
            positive_pair_names.add(pk)
            positive_pair_bvh_names.add(pbk)
            positive_examples.append(
                {
                    "PersonAName": person_a,
                    "PersonBName": person_b,
                    "BVHTypeID": bvh_id,
                    "BVHTypeName": r.get("BVHCT.Description", "") or bvh_id,
                    "HVTI.Score": safe_float(r.get("HVTI.Score")),
                    "HVTI.PotentialLifetimeValueGBP": safe_float(r.get("HVTI.PotentialLifetimeValueGBP")),
                    "HVTI.ExpectedLifetimeValueGBP": safe_float(
                        r.get("HVTI.ExpectedLifetimeValueGBP"),
                        safe_float(r.get("HVTI.PotentialLifetimeValueGBP")),
                    ),
                    "pair_key": pk,
                    "pair_bvh_key": pbk,
                    "raw": r,
                }
            )

    total = max(1, len(rows))
    priors = {}
    avg_score = {}
    avg_value = {}

    for bvh_id, items in by_type.items():
        priors[bvh_id] = len(items) / total
        avg_score[bvh_id] = sum(safe_float(x.get("HVTI.Score")) for x in items) / max(1, len(items))
        avg_value[bvh_id] = sum(
            safe_float(x.get("HVTI.PotentialLifetimeValueGBP") or x.get("HVTI.ExpectedLifetimeValueGBP"))
            for x in items
        ) / max(1, len(items))

    return {
        "label_count": len(rows),
        "bvh_type_priors": priors,
        "bvh_type_avg_score": avg_score,
        "bvh_type_avg_value_gbp": avg_value,
        "positive_pair_names": positive_pair_names,
        "positive_pair_bvh_names": positive_pair_bvh_names,
        "positive_examples": positive_examples,
    }
