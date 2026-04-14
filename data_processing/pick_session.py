#!/usr/bin/env python3
"""
pick_session.py — Session Selection for Hybrid Neural Dynamics Project
======================================================================
Uses AllenSDK metadata to select one optimal session from the Allen
Visual Coding Neuropixels dataset.

Selection criteria:
  - session_type: functional_connectivity
  - Must contain units in all four target visual regions:
    VISp, VISl, VISal, VISpm
  - Ranked by unit_count (descending) then probe_count (descending)

Outputs:
  outputs/chosen_session.json — selected session metadata

Usage:
  python pick_session.py              # real mode (requires network)
  python pick_session.py --demo       # demo mode (synthetic, no network)
"""

import json, sys, argparse
import numpy as np
import pandas as pd
from pathlib import Path

# ── Configuration ──────────────────────────────────────────────────
MANIFEST_PATH  = Path(__file__).parent / "allen_cache" / "manifest.json"
OUTPUT_DIR     = Path(__file__).parent / "outputs"
TARGET_REGIONS = ["VISp", "VISl", "VISal", "VISpm"]
SESSION_TYPE   = "functional_connectivity"


def run_real():
    """Real mode: download metadata from Allen Institute API."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("[1/4] Connecting to Allen Ecephys Project Cache...")
    cache = EcephysProjectCache.from_warehouse(manifest=str(MANIFEST_PATH))

    print("[2/4] Loading session table...")
    sessions = cache.get_session_table()
    print(f"       Total sessions available: {len(sessions)}")

    fc_sessions = sessions[sessions["session_type"] == SESSION_TYPE]
    print(f"       Sessions of type '{SESSION_TYPE}': {len(fc_sessions)}")

    print("[3/4] Loading units table (metadata only)...")
    units = cache.get_units()
    print(f"       Total units across all sessions: {len(units)}")

    print(f"[4/4] Filtering for sessions with regions: {TARGET_REGIONS}")
    target_set = set(TARGET_REGIONS)
    candidates = []
    for sid in fc_sessions.index:
        session_units = units[units["ecephys_session_id"] == sid]
        regions_present = set(
            session_units["ecephys_structure_acronym"].unique()
        )
        if target_set.issubset(regions_present):
            candidates.append({
                "session_id":   int(sid),
                "session_type": SESSION_TYPE,
                "unit_count":   int(len(session_units)),
                "probe_count":  int(session_units["ecephys_probe_id"].nunique()),
                "regions":      sorted(regions_present),
            })

    print(f"       Candidate sessions: {len(candidates)}")
    if not candidates:
        print("ERROR: No sessions found matching all criteria.")
        sys.exit(1)

    candidates.sort(
        key=lambda c: (c["unit_count"], c["probe_count"]), reverse=True
    )
    chosen = candidates[0]
    chosen["target_regions"] = sorted(TARGET_REGIONS)
    chosen["mode"] = "real"
    return chosen


def run_demo():
    """Demo mode: return known session metadata (no network needed)."""
    print("[DEMO] Using pre-validated session metadata...")
    print("       (Session 794812542 was previously identified as optimal)")
    chosen = {
        "session_id":     794812542,
        "session_type":   SESSION_TYPE,
        "unit_count":     1005,
        "probe_count":    6,
        "regions":        [
            "CA1", "DG", "LP", "MB", "TH",
            "VISal", "VISl", "VISp", "VISpm", "VISrl",
        ],
        "target_regions": sorted(TARGET_REGIONS),
        "mode":           "demo",
    }
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Select an Allen Neuropixels session")
    parser.add_argument(
        "--demo", action="store_true",
        help="Use pre-validated metadata (no network needed)",
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print("STEP 1 — SESSION SELECTION")
    print("=" * 60)

    chosen = run_demo() if args.demo else run_real()

    # ── Save ──────────────────────────────────────────────────────
    out_path = OUTPUT_DIR / "chosen_session.json"
    with open(out_path, "w") as f:
        json.dump(chosen, f, indent=2)

    print(f"\n✓ CHOSEN SESSION")
    print(f"  session_id  : {chosen['session_id']}")
    print(f"  session_type: {chosen['session_type']}")
    print(f"  unit_count  : {chosen['unit_count']}")
    print(f"  probe_count : {chosen['probe_count']}")
    print(f"  all regions : {chosen['regions']}")
    print(f"  targets     : {chosen['target_regions']}")
    print(f"  mode        : {chosen['mode']}")
    print(f"\n  Saved → {out_path}")
    return chosen


if __name__ == "__main__":
    main()
