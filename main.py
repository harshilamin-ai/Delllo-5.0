#!/usr/bin/env python3
"""
RAIN5 — main.py

Thin CLI entry point. All the actual matching logic lives in
RAIN_5_Code/rain5.py (run_rain5()) plus its supporting modules — this file
does nothing but parse arguments and call it, then make sure the output ends
up where the calling app service expects to find it.

This file was missing from the delivered RAIN_5_Code.zip; nothing else here
is invented — every flag below maps directly to a real run_rain5() parameter.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from RAIN_5_Code.rain5 import run_rain5  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RAIN5 matching engine")
    p.add_argument("--input-dir", required=True)
    p.add_argument("--knowledge-dir", default=str(ROOT / "Knowledge"))
    p.add_argument("--output-dir", default=str(ROOT / "OUTPUT"))

    p.add_argument("--max-suggestions", type=int, default=250)
    p.add_argument("--min-hvti-score", type=float, default=50.0)
    p.add_argument("--min-usd-value", type=float, default=0.0)
    p.add_argument("--max-pairs", type=int, default=None)
    p.add_argument("--profile-sample-pct", type=float, default=100.0)

    p.add_argument("--no-detail-json", action="store_true")
    p.add_argument("--max-bvh-per-pair", type=int, default=1)

    p.add_argument("--auto-top-k", action="store_true")
    p.add_argument("--target-pair-precision", type=float, default=50.0)
    p.add_argument("--auto-top-k-min-suggestions", type=int, default=10)

    p.add_argument("--bvh-selection-mode", default="canonical_top1")
    p.add_argument("--bvh-top2-margin-threshold", type=float, default=8.0)
    p.add_argument("--bvh-top2-min-canonical-score", type=float, default=65.0)

    p.add_argument("--bvh-recall-floor-spec", default="")
    p.add_argument("--bvh-recall-boost-spec", default="")

    p.add_argument("--no-supervised-weight-tuning", action="store_true")
    p.add_argument("--no-supervised-ranker-v2", action="store_true")
    p.add_argument("--no-prototype-antigeneric", action="store_true")
    p.add_argument("--no-person-saturation-penalty", action="store_true")
    p.add_argument("--no-bvh-precision1", action="store_true")
    p.add_argument("--no-bvh-recall-balancer", action="store_true")

    return p


def main() -> int:
    args = build_parser().parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_absolute():
        input_dir = Path.cwd() / input_dir
    if not input_dir.exists():
        print(f"ERROR: input dir does not exist: {input_dir}", file=sys.stderr)
        return 1

    knowledge_dir = Path(args.knowledge_dir)
    if not knowledge_dir.exists():
        print(f"ERROR: knowledge dir does not exist: {knowledge_dir}", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    kwargs = dict(
        max_suggestions=args.max_suggestions,
        min_hvti_score=args.min_hvti_score,
        min_usd_value=args.min_usd_value,
        max_pairs=args.max_pairs,
        profile_sample_pct=args.profile_sample_pct,
        write_detail_json=not args.no_detail_json,
        max_bvh_per_pair=args.max_bvh_per_pair,
        auto_top_k=args.auto_top_k,
        target_pair_precision=args.target_pair_precision,
        auto_top_k_min_suggestions=args.auto_top_k_min_suggestions,
        bvh_selection_mode=args.bvh_selection_mode,
        bvh_top2_margin_threshold=args.bvh_top2_margin_threshold,
        bvh_top2_min_canonical_score=args.bvh_top2_min_canonical_score,
        supervised_weight_tuning=not args.no_supervised_weight_tuning,
        supervised_ranker_v2=not args.no_supervised_ranker_v2,
        prototype_antigeneric=not args.no_prototype_antigeneric,
        person_saturation_penalty=not args.no_person_saturation_penalty,
        bvh_precision1=not args.no_bvh_precision1,
        bvh_recall_balancer=not args.no_bvh_recall_balancer,
    )
    if args.bvh_recall_floor_spec:
        kwargs["bvh_recall_floor_spec"] = args.bvh_recall_floor_spec
    if args.bvh_recall_boost_spec:
        kwargs["bvh_recall_boost_spec"] = args.bvh_recall_boost_spec

    print(f"[main] input_dir={input_dir}")
    print(f"[main] knowledge_dir={knowledge_dir}")
    print(f"[main] output_dir={output_dir}")

    result = run_rain5(
        input_dir=input_dir,
        output_dir=output_dir,
        knowledge_dir=knowledge_dir,
        **kwargs,
    )

    suggestions_path = result.get("matchSuggestions")
    print(f"[main] matchSuggestions written to: {suggestions_path}")

    # The app service wrapper's file-finder searches for matchSuggestions.JSON
    # anywhere under RAIN5_ROOT/OUTPUT (recursively) — run_rain5() already
    # writes it under output_dir/OUTPUT_<stamp>/, which satisfies that as-is.
    # This copy is a safety net only, in case the wrapper's OUTPUT root
    # differs from --output-dir on some future deployment.
    default_output_root = ROOT / "OUTPUT"
    if suggestions_path and output_dir.resolve() != default_output_root.resolve():
        default_output_root.mkdir(parents=True, exist_ok=True)
        shutil.copy(suggestions_path, default_output_root / "matchSuggestions.JSON")

    return 0


if __name__ == "__main__":
    sys.exit(main())
