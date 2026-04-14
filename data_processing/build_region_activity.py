#!/usr/bin/env python3
"""
build_region_activity.py — Build Region-Level Activity Traces
==============================================================
Loads spike times for filtered units, bins them into time windows,
averages across units within each region, and produces the
observation matrix Y ∈ ℝ^{time × regions}.

Pipeline:
  1. Load spike times for each filtered unit
  2. Bin into time windows (default 20 ms)
  3. Compute firing rate per unit per bin
  4. Average across units within each region → regional activity
  5. Save as region_activity.npz

Inputs:
  outputs/chosen_session.json
  outputs/units_filtered.csv

Outputs:
  outputs/region_activity.npz   — arrays: Y, time_bins, regions, bin_size_s
  outputs/region_activity_summary.json — metadata and statistics

Usage:
  python build_region_activity.py               # real mode
  python build_region_activity.py --demo        # demo mode
  python build_region_activity.py --bin-ms 20   # set bin width
"""

import json, argparse
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR    = Path(__file__).parent / "outputs"
MANIFEST_PATH = Path(__file__).parent / "allen_cache" / "manifest.json"
TARGET_REGIONS = ["VISp", "VISl", "VISal", "VISpm"]
DEFAULT_BIN_MS = 20  # 20 ms bins


def get_spike_times_real(session_id, unit_ids):
    """
    Real mode: load spike times from Allen NWB file.

    Reads directly from the HDF5/NWB file using h5py to avoid the
    AllenSDK's bulk loader, which tries to load ALL spike times (~110M
    floats, ~837 MB) into one array and can cause MemoryError.

    Instead, we read spike times one unit at a time using the ragged
    array index stored in the NWB file.
    """
    import h5py
    from allensdk.brain_observatory.ecephys.ecephys_project_cache import (
        EcephysProjectCache,
    )
    cache = EcephysProjectCache.from_warehouse(manifest=str(MANIFEST_PATH))

    # Find the NWB file in the cache directory.
    # The AllenSDK stores it as: allen_cache/session_<id>/session_<id>.nwb
    cache_dir = MANIFEST_PATH.parent
    nwb_path = cache_dir / f"session_{session_id}" / f"session_{session_id}.nwb"

    if not nwb_path.exists():
        # Also check for the file pattern used by some AllenSDK versions
        candidates = list(cache_dir.rglob(f"*{session_id}*.nwb"))
        if candidates:
            nwb_path = candidates[0]
        else:
            print(f"  NWB file not found in cache. Downloading via AllenSDK...")
            print(f"  (This downloads ~2.5 GB on first run)")
            # This will download the file; it may OOM when accessing
            # spike_times, but the file will be saved to disk for next run.
            try:
                session = cache.get_session_data(session_id)
                spike_times = {}
                for uid in unit_ids:
                    spike_times[uid] = session.spike_times[uid]
                all_spikes = np.concatenate(list(spike_times.values()))
                return spike_times, 0.0, float(np.max(all_spikes)) + 0.001
            except MemoryError:
                print("  MemoryError during bulk load. Retrying with direct HDF5 read...")
                # The file should now be on disk even if spike_times failed
                candidates = list(cache_dir.rglob(f"*{session_id}*.nwb"))
                if candidates:
                    nwb_path = candidates[0]
                else:
                    raise RuntimeError(
                        f"NWB file for session {session_id} not found after download. "
                        f"Looked in: {cache_dir}"
                    )

    print(f"  Reading spike times directly from NWB file...")
    print(f"  NWB path: {nwb_path}")

    # ── Read spike times from HDF5 in a memory-efficient way ─────
    # NWB stores spike times as a ragged array:
    #   /units/spike_times      — flat array of ALL spike times
    #   /units/spike_times_index — end-index for each unit
    #   /units/id               — unit IDs
    spike_times = {}
    t_max = 0.0

    with h5py.File(nwb_path, "r") as f:
        all_spike_data = f["units"]["spike_times"]
        spike_index    = f["units"]["spike_times_index"][:]
        unit_id_array  = f["units"]["id"][:]

        # Build a lookup: Allen unit_id → row index in NWB table
        uid_to_row = {int(uid): row for row, uid in enumerate(unit_id_array)}

        loaded = 0
        skipped = 0
        for uid in unit_ids:
            if uid not in uid_to_row:
                skipped += 1
                continue
            row = uid_to_row[uid]
            start_idx = int(spike_index[row - 1]) if row > 0 else 0
            end_idx   = int(spike_index[row])
            spikes = all_spike_data[start_idx:end_idx]
            spike_times[uid] = np.array(spikes)
            if len(spikes) > 0:
                t_max = max(t_max, float(spikes[-1]))
            loaded += 1

        if loaded % 50 == 0 or loaded == len(unit_ids):
            pass  # progress is shown after the loop

    print(f"  Loaded: {loaded} units  |  Skipped: {skipped}")
    t_start = 0.0
    t_end = t_max + 0.001
    return spike_times, t_start, t_end


def get_spike_times_demo(session_id, units_df):
    """Demo mode: generate realistic synthetic spike trains."""
    print("  [DEMO] Generating synthetic spike trains...")
    np.random.seed(42)

    # Simulate a 30-minute recording session (realistic for Allen data)
    t_start = 0.0
    t_end = 1800.0  # 30 minutes in seconds

    spike_times = {}
    for idx, row in units_df.iterrows():
        # Use firing_rate from the units table if available, else default
        fr = row.get("firing_rate", 5.0)
        # Generate Poisson spike train
        n_expected = int(fr * (t_end - t_start))
        isi = np.random.exponential(1.0 / max(fr, 0.1), n_expected)
        times = np.cumsum(isi)
        times = times[times < t_end]
        spike_times[idx] = times

    return spike_times, t_start, t_end


def bin_spikes(spike_times_dict, t_start, t_end, bin_size_s):
    """
    Bin spike times into fixed-width time windows.
    Returns: (n_bins,) count array per unit, plus bin edges.
    """
    bin_edges = np.arange(t_start, t_end + bin_size_s, bin_size_s)
    n_bins = len(bin_edges) - 1
    binned = {}
    for uid, spikes in spike_times_dict.items():
        counts, _ = np.histogram(spikes, bins=bin_edges)
        binned[uid] = counts
    # Time centers
    time_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return binned, time_centers, bin_edges


def compute_region_activity(binned, units_df, bin_size_s):
    """
    Average binned spike counts across units within each region.
    Converts to firing rate (spikes/second) by dividing by bin width.
    Returns: Y array (n_bins × n_regions), region order list
    """
    regions_sorted = sorted(TARGET_REGIONS)
    n_bins = len(next(iter(binned.values())))

    Y = np.zeros((n_bins, len(regions_sorted)))

    for i, region in enumerate(regions_sorted):
        region_unit_ids = units_df[
            units_df["ecephys_structure_acronym"] == region
        ].index.tolist()
        if not region_unit_ids:
            print(f"  WARNING: No units found for region {region}")
            continue

        # Stack all unit counts for this region
        region_matrix = np.array([binned[uid] for uid in region_unit_ids if uid in binned])
        # Average firing rate = mean count / bin_size
        Y[:, i] = region_matrix.mean(axis=0) / bin_size_s

    return Y, regions_sorted


def main():
    parser = argparse.ArgumentParser(
        description="Build region-level activity traces"
    )
    parser.add_argument("--demo", action="store_true")
    parser.add_argument(
        "--bin-ms", type=float, default=DEFAULT_BIN_MS,
        help=f"Bin width in milliseconds (default {DEFAULT_BIN_MS})",
    )
    args = parser.parse_args()
    bin_size_s = args.bin_ms / 1000.0

    print("=" * 60)
    print("STEP 3 — BUILD REGION-LEVEL ACTIVITY TRACES")
    print("=" * 60)

    # ── Load inputs ──────────────────────────────────────────────
    session_path = OUTPUT_DIR / "chosen_session.json"
    units_path   = OUTPUT_DIR / "units_filtered.csv"
    for p in [session_path, units_path]:
        if not p.exists():
            print(f"ERROR: {p} not found. Run earlier scripts first.")
            return

    with open(session_path) as f:
        session_info = json.load(f)
    sid = session_info["session_id"]

    units_df = pd.read_csv(units_path, index_col=0)
    print(f"\nSession: {sid}")
    print(f"Filtered units loaded: {len(units_df)}")
    print(f"Bin size: {args.bin_ms} ms ({bin_size_s} s)")

    # ── Load spike times ─────────────────────────────────────────
    print(f"\n[1/3] Loading spike times...")
    if args.demo:
        spike_times, t_start, t_end = get_spike_times_demo(sid, units_df)
    else:
        unit_ids = units_df.index.tolist()
        spike_times, t_start, t_end = get_spike_times_real(sid, unit_ids)

    duration = t_end - t_start
    total_spikes = sum(len(s) for s in spike_times.values())
    print(f"       Duration: {duration:.1f} s ({duration/60:.1f} min)")
    print(f"       Total spikes: {total_spikes:,}")
    print(f"       Units with spikes: {len(spike_times)}")

    # ── Bin spikes ───────────────────────────────────────────────
    print(f"\n[2/3] Binning spikes into {args.bin_ms} ms windows...")
    binned, time_centers, bin_edges = bin_spikes(
        spike_times, t_start, t_end, bin_size_s
    )
    n_bins = len(time_centers)
    print(f"       Number of time bins: {n_bins:,}")

    # ── Compute regional activity ────────────────────────────────
    print(f"\n[3/3] Averaging across units per region...")
    Y, regions = compute_region_activity(binned, units_df, bin_size_s)

    print(f"\n  Y shape: {Y.shape}  (time_bins × regions)")
    print(f"  Regions: {regions}")
    print(f"  Activity statistics (firing rate, Hz):")
    for i, r in enumerate(regions):
        col = Y[:, i]
        print(f"    {r}: mean={col.mean():.2f}, std={col.std():.2f}, "
              f"min={col.min():.2f}, max={col.max():.2f}")

    # ── Save outputs ─────────────────────────────────────────────
    npz_path = OUTPUT_DIR / "region_activity.npz"
    np.savez(
        npz_path,
        Y=Y,
        time_bins=time_centers,
        bin_edges=bin_edges,
        regions=np.array(regions),
        bin_size_s=np.array(bin_size_s),
    )
    print(f"\n  Saved activity data → {npz_path}")

    summary = {
        "session_id":       sid,
        "bin_size_ms":      args.bin_ms,
        "bin_size_s":       bin_size_s,
        "n_time_bins":      int(n_bins),
        "n_regions":        len(regions),
        "regions":          regions,
        "duration_s":       float(duration),
        "total_spikes":     int(total_spikes),
        "n_units_used":     int(len(spike_times)),
        "Y_shape":          list(Y.shape),
        "per_region_stats": {
            r: {
                "mean_hz": float(np.mean(Y[:, i])),
                "std_hz":  float(np.std(Y[:, i])),
                "min_hz":  float(np.min(Y[:, i])),
                "max_hz":  float(np.max(Y[:, i])),
                "n_units": int(
                    (units_df["ecephys_structure_acronym"] == r).sum()
                ),
            }
            for i, r in enumerate(regions)
        },
        "mode": "demo" if args.demo else "real",
    }
    summary_path = OUTPUT_DIR / "region_activity_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary       → {summary_path}")

    print(f"\n✓ Region activity matrix Y ready for modeling.")
    return Y, time_centers, regions


if __name__ == "__main__":
    main()
