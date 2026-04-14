#!/usr/bin/env python3
"""
build_rich_stimulus.py — Rich Parametric Stimulus Encoding
===========================================================
Extracts detailed stimulus parameters from Allen SDK instead of binary one-hot.

For each stimulus type, we extract continuous features:
  - Drifting gratings: orientation (sin/cos), temporal_freq, spatial_freq, contrast
  - Static gratings: orientation (sin/cos), spatial_freq, contrast, phase
  - Natural scenes: image embedding (PCA of image or index encoding)
  - Natural movies: frame number (normalized), clip identity
  - Gabors: x, y, orientation, spatial_freq, contrast
  - Flashes: intensity (+1 white, -1 black, 0 off)

Output: region_activity_with_rich_stim.npz
  - U_rich: (T, D) continuous stimulus features
  - U_binary: (T, 7) original binary encoding for comparison
  - feature_names: list of feature names
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

OUTPUT_DIR = Path(__file__).parent / "outputs"
MANIFEST_PATH = Path(__file__).parent / "allen_cache" / "manifest.json"


def encode_orientation(degrees):
    """Encode orientation as sin/cos (handles circularity)."""
    rad = np.deg2rad(degrees * 2)  # *2 because orientation is 180° periodic
    return np.sin(rad), np.cos(rad)


def build_rich_stimulus_matrix(stim_df, time_bins, stimulus_names):
    """
    Build rich continuous stimulus encoding from Allen stimulus_presentations.

    Returns:
        U_rich: (T, D) array of continuous features
        feature_names: list of feature names
    """
    T = len(time_bins)

    # Initialize feature columns
    features = {
        # Stimulus type indicators (keep for interpretability)
        'stim_drifting_grating': np.zeros(T),
        'stim_static_grating': np.zeros(T),
        'stim_natural_scene': np.zeros(T),
        'stim_natural_movie': np.zeros(T),
        'stim_gabor': np.zeros(T),
        'stim_flash': np.zeros(T),
        'stim_other': np.zeros(T),

        # Orientation (sin/cos encoding, shared across grating types)
        'ori_sin': np.zeros(T),
        'ori_cos': np.zeros(T),

        # Temporal frequency (Hz, normalized)
        'temporal_freq': np.zeros(T),

        # Spatial frequency (cycles/deg, normalized)
        'spatial_freq': np.zeros(T),

        # Contrast (0-1)
        'contrast': np.zeros(T),

        # Phase (for static gratings, sin/cos encoded)
        'phase_sin': np.zeros(T),
        'phase_cos': np.zeros(T),

        # Natural scene index (normalized 0-1)
        'scene_index': np.zeros(T),

        # Natural movie features
        'movie_frame': np.zeros(T),  # normalized frame within clip
        'movie_clip_1': np.zeros(T),  # one-hot for different clips
        'movie_clip_2': np.zeros(T),
        'movie_clip_3': np.zeros(T),

        # Gabor position (normalized to screen)
        'gabor_x': np.zeros(T),
        'gabor_y': np.zeros(T),

        # Flash intensity
        'flash_intensity': np.zeros(T),
    }

    # Normalization constants (from Allen documentation)
    MAX_TEMPORAL_FREQ = 15.0  # Hz
    MAX_SPATIAL_FREQ = 0.32   # cycles/degree
    MAX_SCENE_INDEX = 117.0   # natural scenes
    MAX_MOVIE_FRAMES = 900.0  # approximate

    print(f"Processing {len(stim_df)} stimulus presentations...")
    print(f"Columns available: {stim_df.columns.tolist()[:20]}...")  # First 20

    # Check what columns we have
    has_orientation = 'orientation' in stim_df.columns
    has_temporal_freq = 'temporal_frequency' in stim_df.columns
    has_spatial_freq = 'spatial_frequency' in stim_df.columns
    has_contrast = 'contrast' in stim_df.columns
    has_phase = 'phase' in stim_df.columns
    has_frame = 'frame' in stim_df.columns
    has_x_position = 'x_position' in stim_df.columns
    has_y_position = 'y_position' in stim_df.columns
    has_color = 'color' in stim_df.columns

    print(f"Available features: orientation={has_orientation}, temporal_freq={has_temporal_freq}, "
          f"spatial_freq={has_spatial_freq}, contrast={has_contrast}, phase={has_phase}, "
          f"frame={has_frame}, position={has_x_position}")

    for _, row in stim_df.iterrows():
        start_t = row['start_time']
        stop_t = row['stop_time']
        stim_name = row['stimulus_name'].lower() if pd.notna(row['stimulus_name']) else ''

        # Find bins that fall within this presentation
        start_bin = np.searchsorted(time_bins, start_t)
        stop_bin = np.searchsorted(time_bins, stop_t)

        if start_bin >= T or stop_bin > T or start_bin >= stop_bin:
            continue

        sl = slice(start_bin, stop_bin)

        # Skip spontaneous activity
        if 'spontaneous' in stim_name:
            continue

        # --- Drifting Gratings ---
        if 'drifting' in stim_name and 'grating' in stim_name:
            features['stim_drifting_grating'][sl] = 1.0

            if has_orientation and pd.notna(row.get('orientation')):
                ori = float(row['orientation'])
                s, c = encode_orientation(ori)
                features['ori_sin'][sl] = s
                features['ori_cos'][sl] = c

            if has_temporal_freq and pd.notna(row.get('temporal_frequency')):
                tf = float(row['temporal_frequency'])
                features['temporal_freq'][sl] = tf / MAX_TEMPORAL_FREQ

            if has_spatial_freq and pd.notna(row.get('spatial_frequency')):
                sf = float(row['spatial_frequency'])
                features['spatial_freq'][sl] = sf / MAX_SPATIAL_FREQ

            if has_contrast and pd.notna(row.get('contrast')):
                features['contrast'][sl] = float(row['contrast'])

        # --- Static Gratings ---
        elif 'static' in stim_name and 'grating' in stim_name:
            features['stim_static_grating'][sl] = 1.0

            if has_orientation and pd.notna(row.get('orientation')):
                ori = float(row['orientation'])
                s, c = encode_orientation(ori)
                features['ori_sin'][sl] = s
                features['ori_cos'][sl] = c

            if has_spatial_freq and pd.notna(row.get('spatial_frequency')):
                sf = float(row['spatial_frequency'])
                features['spatial_freq'][sl] = sf / MAX_SPATIAL_FREQ

            if has_contrast and pd.notna(row.get('contrast')):
                features['contrast'][sl] = float(row['contrast'])

            if has_phase and pd.notna(row.get('phase')):
                phase = float(row['phase'])
                features['phase_sin'][sl] = np.sin(np.deg2rad(phase))
                features['phase_cos'][sl] = np.cos(np.deg2rad(phase))

        # --- Natural Scenes ---
        elif 'natural' in stim_name and 'scene' in stim_name:
            features['stim_natural_scene'][sl] = 1.0

            if has_frame and pd.notna(row.get('frame')):
                frame_idx = float(row['frame'])
                features['scene_index'][sl] = frame_idx / MAX_SCENE_INDEX

        # --- Natural Movies ---
        elif 'natural' in stim_name and 'movie' in stim_name:
            features['stim_natural_movie'][sl] = 1.0

            if has_frame and pd.notna(row.get('frame')):
                frame_num = float(row['frame'])
                features['movie_frame'][sl] = frame_num / MAX_MOVIE_FRAMES

            # Identify which clip
            if 'one' in stim_name or '1' in stim_name:
                features['movie_clip_1'][sl] = 1.0
            elif 'two' in stim_name or '2' in stim_name:
                features['movie_clip_2'][sl] = 1.0
            elif 'three' in stim_name or '3' in stim_name:
                features['movie_clip_3'][sl] = 1.0

        # --- Gabors ---
        elif 'gabor' in stim_name:
            features['stim_gabor'][sl] = 1.0

            if has_orientation and pd.notna(row.get('orientation')):
                ori = float(row['orientation'])
                s, c = encode_orientation(ori)
                features['ori_sin'][sl] = s
                features['ori_cos'][sl] = c

            if has_spatial_freq and pd.notna(row.get('spatial_frequency')):
                sf = float(row['spatial_frequency'])
                features['spatial_freq'][sl] = sf / MAX_SPATIAL_FREQ

            if has_contrast and pd.notna(row.get('contrast')):
                features['contrast'][sl] = float(row['contrast'])

            if has_x_position and pd.notna(row.get('x_position')):
                # Normalize to [-1, 1] assuming screen coords
                features['gabor_x'][sl] = float(row['x_position']) / 60.0  # approximate

            if has_y_position and pd.notna(row.get('y_position')):
                features['gabor_y'][sl] = float(row['y_position']) / 60.0

        # --- Flashes ---
        elif 'flash' in stim_name:
            features['stim_flash'][sl] = 1.0

            if has_color and pd.notna(row.get('color')):
                color = row['color']
                if color == 1.0 or color == 'white':
                    features['flash_intensity'][sl] = 1.0
                elif color == -1.0 or color == 'black':
                    features['flash_intensity'][sl] = -1.0

        # --- Other stimuli ---
        else:
            features['stim_other'][sl] = 1.0

    # Stack into matrix
    feature_names = list(features.keys())
    U_rich = np.column_stack([features[name] for name in feature_names])

    # Remove all-zero columns
    nonzero_mask = np.any(U_rich != 0, axis=0)
    U_rich = U_rich[:, nonzero_mask]
    feature_names = [name for name, keep in zip(feature_names, nonzero_mask) if keep]

    print(f"Rich stimulus matrix: {U_rich.shape} ({len(feature_names)} non-zero features)")
    print(f"Features: {feature_names}")

    return U_rich.astype(np.float32), feature_names


def build_binary_stimulus_matrix(stim_df, time_bins, stimulus_names):
    """Original binary one-hot encoding for comparison."""
    T = len(time_bins)
    active_stimuli = [s for s in stimulus_names if s != 'spontaneous']
    n_stimuli = len(active_stimuli)
    U_binary = np.zeros((T, n_stimuli), dtype=np.float32)

    for stim_name in active_stimuli:
        stim_idx = active_stimuli.index(stim_name)
        events = stim_df[stim_df['stimulus_name'] == stim_name]

        for _, row in events.iterrows():
            start_t = row['start_time']
            stop_t = row['stop_time']
            start_bin = np.searchsorted(time_bins, start_t)
            stop_bin = np.searchsorted(time_bins, stop_t)

            if start_bin < T and stop_bin <= T and start_bin < stop_bin:
                U_binary[start_bin:stop_bin, stim_idx] = 1.0

    return U_binary, active_stimuli


def main():
    print("=" * 60)
    print("BUILDING RICH STIMULUS ENCODING")
    print("=" * 60)

    # Load existing activity matrix
    npz_path = OUTPUT_DIR / "region_activity_with_stim.npz"
    if not npz_path.exists():
        npz_path = OUTPUT_DIR / "region_activity.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Cannot find activity file. Run build_region_activity.py first.")

    data = np.load(npz_path)
    Y = data['Y']
    time_bins = data['time_bins']
    regions = data['regions']

    # Get session ID
    summary_path = OUTPUT_DIR / "region_activity_summary.json"
    with open(summary_path, 'r') as f:
        summary = json.load(f)
    session_id = summary['session_id']

    print(f"Session: {session_id}")
    print(f"Activity shape: {Y.shape}")
    print(f"Time bins: {len(time_bins)}")

    # Load Allen session
    print("\nLoading Allen session data...")
    cache = EcephysProjectCache.from_warehouse(manifest=str(MANIFEST_PATH))
    session = cache.get_session_data(session_id)

    stim_df = session.stimulus_presentations
    stimulus_names = list(session.stimulus_names)

    print(f"Stimulus presentations: {len(stim_df)}")
    print(f"Stimulus types: {stimulus_names}")

    # Show sample of stimulus parameters
    print("\nSample stimulus presentation:")
    print(stim_df.iloc[0].to_dict())

    # Build rich encoding
    print("\n" + "-" * 40)
    print("Building RICH stimulus encoding...")
    U_rich, feature_names = build_rich_stimulus_matrix(stim_df, time_bins, stimulus_names)

    # Build binary encoding for comparison
    print("\n" + "-" * 40)
    print("Building BINARY stimulus encoding (for comparison)...")
    U_binary, binary_stim_names = build_binary_stimulus_matrix(stim_df, time_bins, stimulus_names)

    # Statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Rich encoding:   {U_rich.shape} ({len(feature_names)} features)")
    print(f"Binary encoding: {U_binary.shape} ({len(binary_stim_names)} features)")
    print(f"\nRich feature statistics:")
    for i, name in enumerate(feature_names):
        col = U_rich[:, i]
        nonzero = np.sum(col != 0)
        print(f"  {name:25s}: mean={col.mean():.3f}, std={col.std():.3f}, "
              f"nonzero={nonzero/len(col)*100:.1f}%")

    # Save
    out_path = OUTPUT_DIR / "region_activity_with_rich_stim.npz"
    np.savez(
        out_path,
        Y=Y,
        U_rich=U_rich,
        U_binary=U_binary,
        time_bins=time_bins,
        bin_edges=data['bin_edges'],
        regions=regions,
        feature_names=np.array(feature_names),
        binary_stim_names=np.array(binary_stim_names),
        bin_size_s=data['bin_size_s'],
    )

    print(f"\nSaved to: {out_path}")
    print(f"  Y.shape:        {Y.shape}")
    print(f"  U_rich.shape:   {U_rich.shape}")
    print(f"  U_binary.shape: {U_binary.shape}")
    print(f"  feature_names:  {feature_names}")

    return U_rich, feature_names


if __name__ == "__main__":
    main()
