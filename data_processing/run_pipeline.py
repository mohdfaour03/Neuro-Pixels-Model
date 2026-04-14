#!/usr/bin/env python3
"""
run_pipeline.py — Master Pipeline Script
==========================================
Runs the full data processing pipeline in sequence:
  Step 1: pick_session.py   → chosen_session.json
  Step 2: inspect_units.py  → units_filtered.csv, units_summary.json
  Step 3: build_region_activity.py → region_activity.npz, region_activity_summary.json

Usage:
  python run_pipeline.py              # real mode (requires AllenSDK + network)
  python run_pipeline.py --demo       # demo mode (synthetic data, no network)
  python run_pipeline.py --demo --bin-ms 10   # demo with 10ms bins
"""

import subprocess, sys, argparse
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PYTHON     = sys.executable  # use the same Python interpreter


def run_step(name, script, extra_args=None):
    """Run a pipeline step as a subprocess."""
    cmd = [PYTHON, str(SCRIPT_DIR / script)]
    if extra_args:
        cmd.extend(extra_args)

    print(f"\n{'━' * 60}")
    print(f"  Running: {' '.join(cmd)}")
    print(f"{'━' * 60}\n")

    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR))
    if result.returncode != 0:
        print(f"\n✗ {name} FAILED (exit code {result.returncode})")
        sys.exit(result.returncode)
    print(f"\n✓ {name} completed successfully.")


def main():
    parser = argparse.ArgumentParser(description="Run full data processing pipeline")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic data (no network needed)")
    parser.add_argument("--bin-ms", type=float, default=20,
                        help="Bin width in milliseconds (default 20)")
    args = parser.parse_args()

    mode_args = ["--demo"] if args.demo else []

    print("╔" + "═" * 58 + "╗")
    print("║  DATA PROCESSING PIPELINE                                ║")
    print("║  Hybrid Neural Dynamics Project — EECE 798K              ║")
    print("╠" + "═" * 58 + "╣")
    mode = "DEMO (synthetic)" if args.demo else "REAL (Allen API)"
    print(f"║  Mode: {mode:<50}║")
    print(f"║  Bin size: {args.bin_ms} ms{' ' * (46 - len(str(args.bin_ms)))}║")
    print("╚" + "═" * 58 + "╝")

    # Step 1 — Session Selection
    run_step("Step 1: Session Selection", "pick_session.py", mode_args)

    # Step 2 — Unit Filtering
    run_step("Step 2: Unit Filtering", "inspect_units.py", mode_args)

    # Step 3 — Region Activity
    step3_args = mode_args + ["--bin-ms", str(args.bin_ms)]
    run_step("Step 3: Region Activity", "build_region_activity.py", step3_args)

    # ── Final summary ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    output_dir = SCRIPT_DIR / "outputs"
    print(f"\nAll outputs in: {output_dir}/")
    for f in sorted(output_dir.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            if size > 1024 * 1024:
                size_str = f"{size / (1024*1024):.1f} MB"
            elif size > 1024:
                size_str = f"{size / 1024:.1f} KB"
            else:
                size_str = f"{size} B"
            print(f"  {f.name:<35} {size_str:>10}")

    print("\nNext step: use region_activity.npz as input for Wilson-Cowan model.")


if __name__ == "__main__":
    main()
