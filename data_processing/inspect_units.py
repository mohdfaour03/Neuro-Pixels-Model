#!/usr/bin/env python3
"""
inspect_units.py — Unit Inspection & Quality Filtering
=======================================================
Loads the chosen session, restricts to the four target visual regions,
applies quality filters, and exports a clean units table.

Quality filters (standard Allen thresholds):
  - presence_ratio  > 0.9   (unit recorded in >90% of session)
  - isi_violations  < 0.5   (low inter-spike-interval violations)
  - amplitude_cutoff < 0.1  (stable amplitude throughout recording)

Inputs:
  outputs/chosen_session.json

Outputs:
  outputs/units_filtered.csv     — clean units table
  outputs/units_summary.json     — counts per region, before/after filtering

Usage:
  python inspect_units.py              # real mode
  python inspect_units.py --demo       # demo mode (synthetic units)
"""

import json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR     = Path(__file__).parent / "outputs"
MANIFEST_PATH  = Path(__file__).parent / "allen_cache" / "manifest.json"
TARGET_REGIONS = ["VISp", "VISl", "VISal", "VISpm"]

# Quality thresholds
PRESENCE_RATIO_MIN  = 0.9
ISI_VIOLATIONS_MAX  = 0.5
AMPLITUDE_CUTOFF_MAX = 0.1


def run_real(session_info):
    """Real mode: load units from Allen cache."""
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )

    sid = session_info["session_id"]
    cache = EcephysProjectCache.from_warehouse(manifest=str(MANIFEST_PATH))

    print(f"[1/3] Loading units for session {sid}...")
    units = cache.get_units()
    session_units = units[units["ecephys_session_id"] == sid].copy()
    print(f"       Total units in session: {len(session_units)}")
    return session_units


def run_demo(session_info):
    """Demo mode: generate realistic synthetic units table."""
    print("[DEMO] Generating synthetic units table...")
    np.random.seed(42)
    n_units = session_info["unit_count"]  # 1005

    # Distribute units across regions (realistic proportions)
    all_regions = session_info["regions"]
    # VIS regions get more units; others get fewer
    vis_regions = [r for r in all_regions if r.startswith("VIS")]
    other_regions = [r for r in all_regions if not r.startswith("VIS")]

    region_assignments = []
    # ~60% in VIS regions, ~40% in others
    n_vis = int(n_units * 0.6)
    n_other = n_units - n_vis
    for _ in range(n_vis):
        region_assignments.append(np.random.choice(vis_regions))
    for _ in range(n_other):
        region_assignments.append(np.random.choice(other_regions))
    np.random.shuffle(region_assignments)

    units = pd.DataFrame({
        "ecephys_session_id":         session_info["session_id"],
        "ecephys_structure_acronym":  region_assignments,
        "ecephys_probe_id":           np.random.choice(
            range(100000, 100006), size=n_units
        ),
        "presence_ratio":  np.clip(np.random.beta(8, 1.5, n_units), 0, 1),
        "isi_violations":  np.clip(np.random.exponential(0.15, n_units), 0, 5),
        "amplitude_cutoff": np.clip(np.random.exponential(0.04, n_units), 0, 0.5),
        "firing_rate":     np.clip(np.random.lognormal(1.5, 1.0, n_units), 0.01, 100),
        "snr":             np.clip(np.random.lognormal(0.5, 0.6, n_units), 0.1, 20),
    })
    units.index.name = "unit_id"
    print(f"       Synthetic units generated: {len(units)}")
    return units


def main():
    parser = argparse.ArgumentParser(description="Filter units by region and quality")
    parser.add_argument("--demo", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("STEP 2 — UNIT INSPECTION & QUALITY FILTERING")
    print("=" * 60)

    # ── Load session info ────────────────────────────────────────
    session_path = OUTPUT_DIR / "chosen_session.json"
    if not session_path.exists():
        print(f"ERROR: {session_path} not found. Run pick_session.py first.")
        return
    with open(session_path) as f:
        session_info = json.load(f)
    print(f"\nSession: {session_info['session_id']}")

    # ── Get units ────────────────────────────────────────────────
    units = run_demo(session_info) if args.demo else run_real(session_info)

    # ── Restrict to target regions ───────────────────────────────
    print(f"\n[2/3] Restricting to target regions: {TARGET_REGIONS}")
    target_units = units[
        units["ecephys_structure_acronym"].isin(TARGET_REGIONS)
    ].copy()
    print(f"       Units in target regions: {len(target_units)}")

    before_counts = (
        target_units.groupby("ecephys_structure_acronym").size().to_dict()
    )
    print("       Per region (before filtering):")
    for r in sorted(before_counts):
        print(f"         {r}: {before_counts[r]}")

    # ── Apply quality filters ────────────────────────────────────
    print(f"\n[3/3] Applying quality filters:")
    print(f"       presence_ratio  > {PRESENCE_RATIO_MIN}")
    print(f"       isi_violations  < {ISI_VIOLATIONS_MAX}")
    print(f"       amplitude_cutoff < {AMPLITUDE_CUTOFF_MAX}")

    mask = (
        (target_units["presence_ratio"] > PRESENCE_RATIO_MIN)
        & (target_units["isi_violations"] < ISI_VIOLATIONS_MAX)
        & (target_units["amplitude_cutoff"] < AMPLITUDE_CUTOFF_MAX)
    )
    filtered = target_units[mask].copy()
    n_removed = len(target_units) - len(filtered)
    print(f"       Removed: {n_removed}  |  Kept: {len(filtered)}")

    after_counts = (
        filtered.groupby("ecephys_structure_acronym").size().to_dict()
    )
    print("       Per region (after filtering):")
    for r in sorted(after_counts):
        print(f"         {r}: {after_counts[r]}")

    # ── Save outputs ─────────────────────────────────────────────
    csv_path = OUTPUT_DIR / "units_filtered.csv"
    filtered.to_csv(csv_path)
    print(f"\n  Saved filtered units → {csv_path}")

    summary = {
        "session_id":      session_info["session_id"],
        "target_regions":  TARGET_REGIONS,
        "quality_filters": {
            "presence_ratio_min":   PRESENCE_RATIO_MIN,
            "isi_violations_max":   ISI_VIOLATIONS_MAX,
            "amplitude_cutoff_max": AMPLITUDE_CUTOFF_MAX,
        },
        "total_session_units":     int(len(units)),
        "units_in_target_regions": int(len(target_units)),
        "units_after_filtering":   int(len(filtered)),
        "units_removed":           int(n_removed),
        "before_counts":           before_counts,
        "after_counts":            after_counts,
        "mode": "demo" if args.demo else "real",
    }
    summary_path = OUTPUT_DIR / "units_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary       → {summary_path}")

    return filtered


if __name__ == "__main__":
    main()
