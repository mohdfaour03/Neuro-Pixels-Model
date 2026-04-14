#!/usr/bin/env python3
"""
add_behavioral_data.py — Add Running Speed & Pupil Data to .npz
================================================================
Extracts behavioral signals (running speed, pupil width) from the
NWB file and adds them to region_activity_with_rich_stim.npz.

Behavioral signals are resampled to match the neural data time bins
(20ms) using linear interpolation. NaN values in pupil data are
forward-filled then back-filled.

New keys added to .npz:
  - running_speed: (T,) running speed in cm/s
  - pupil_width:   (T,) pupil width in pixels (proxy for arousal)
  - behavioral_names: ['running_speed', 'pupil_width']

Usage:
  python add_behavioral_data.py
"""

import h5py
import numpy as np
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "outputs"
NWB_PATH = Path(__file__).parent / "allen_cache" / "session_794812542" / "session_794812542.nwb"


def interpolate_to_bins(timestamps, data, target_bins):
    """
    Linearly interpolate a behavioral signal to match neural time bins.
    Handles NaNs by forward-fill then back-fill before interpolation.
    """
    # Remove NaN values for interpolation
    valid_mask = ~np.isnan(data)
    if valid_mask.sum() == 0:
        print("  WARNING: all NaN, returning zeros")
        return np.zeros(len(target_bins))

    valid_ts = timestamps[valid_mask]
    valid_data = data[valid_mask]

    # Interpolate to target time bins
    interpolated = np.interp(target_bins, valid_ts, valid_data)
    return interpolated


def main():
    print("=" * 60)
    print("ADDING BEHAVIORAL DATA TO .npz")
    print("=" * 60)

    # Load existing .npz
    npz_path = OUTPUT_DIR / "region_activity_with_rich_stim.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find {npz_path}. Run build_rich_stimulus.py first.")

    existing = np.load(npz_path, allow_pickle=True)
    time_bins = existing['time_bins']
    print(f"Neural data: {len(time_bins)} time bins")
    print(f"Time range: {time_bins[0]:.1f} - {time_bins[-1]:.1f}s")

    # Check NWB file
    if not NWB_PATH.exists():
        raise FileNotFoundError(f"NWB file not found at {NWB_PATH}")
    print(f"\nNWB file: {NWB_PATH}")

    with h5py.File(NWB_PATH, 'r') as f:

        # ---- Running Speed ----
        print("\n--- Running Speed ---")
        run_data = f['processing/running/running_speed/data'][:]
        run_ts = f['processing/running/running_speed/timestamps'][:]
        print(f"  Raw: {len(run_data)} samples, {run_ts[0]:.1f}-{run_ts[-1]:.1f}s")
        print(f"  NaN: {np.sum(np.isnan(run_data))}")

        running_speed = interpolate_to_bins(run_ts, run_data, time_bins)
        print(f"  Resampled: {len(running_speed)} bins")
        print(f"  Range: [{running_speed.min():.2f}, {running_speed.max():.2f}] cm/s")
        print(f"  Mean: {running_speed.mean():.2f} cm/s")

        # ---- Pupil Width ----
        print("\n--- Pupil Width ---")
        pupil_data = f['processing/eye_tracking/pupil_ellipse_fits/width'][:]
        pupil_ts = f['processing/eye_tracking/pupil_ellipse_fits/timestamps'][:]
        n_nan = np.sum(np.isnan(pupil_data))
        print(f"  Raw: {len(pupil_data)} samples, {pupil_ts[0]:.1f}-{pupil_ts[-1]:.1f}s")
        print(f"  NaN: {n_nan} ({100*n_nan/len(pupil_data):.1f}%)")

        pupil_width = interpolate_to_bins(pupil_ts, pupil_data, time_bins)
        print(f"  Resampled: {len(pupil_width)} bins")
        print(f"  Range: [{pupil_width.min():.2f}, {pupil_width.max():.2f}] px")
        print(f"  Mean: {pupil_width.mean():.2f} px")

    # ---- Save updated .npz ----
    print("\n--- Saving ---")

    # Rebuild with all existing data + new behavioral data
    save_dict = {key: existing[key] for key in existing.files}
    save_dict['running_speed'] = running_speed.astype(np.float32)
    save_dict['pupil_width'] = pupil_width.astype(np.float32)
    save_dict['behavioral_names'] = np.array(['running_speed', 'pupil_width'])

    np.savez(npz_path, **save_dict)

    print(f"Saved to: {npz_path}")
    print(f"New keys: running_speed {running_speed.shape}, pupil_width {pupil_width.shape}")

    # Verify
    print("\n--- Verification ---")
    check = np.load(npz_path, allow_pickle=True)
    print(f"All keys: {list(check.keys())}")
    print(f"running_speed: {check['running_speed'].shape}")
    print(f"pupil_width: {check['pupil_width'].shape}")

    print(f"\n{'='*60}")
    print("DONE — behavioral data added successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
