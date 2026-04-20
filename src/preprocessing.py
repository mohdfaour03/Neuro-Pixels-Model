import numpy as np
from scipy.ndimage import gaussian_filter1d


def collect_multi_region_units(session, candidate_regions, min_units_per_region=10):
    """
    Selects cortical / thalamic regions with enough units for a stable multi-region prototype.

    Returns:
        retained_region_units: dict[str, np.ndarray]
        region_metadata: dict[str, dict]
    """
    units = session.units
    retained_region_units = {}
    region_metadata = {}

    for region in candidate_regions:
        region_units = units[units["ecephys_structure_acronym"] == region]
        unit_ids = region_units.index.values
        unit_count = int(len(unit_ids))

        if unit_count < min_units_per_region:
            print(f"[{region}] Skipping region. Found {unit_count} units (< {min_units_per_region}).")
            continue

        retained_region_units[region] = unit_ids
        region_metadata[region] = {
            "num_units": unit_count,
            "unit_ids": [int(uid) for uid in unit_ids],
        }
        print(f"[{region}] Retained region with {unit_count} units.")

    if not retained_region_units:
        raise ValueError(
            "No selected regions met the minimum unit threshold. "
            "Lower min_units_per_region or choose different regions."
        )

    return retained_region_units, region_metadata


def build_multi_region_population_activity(
    session,
    region_unit_map,
    t_start,
    t_stop,
    bin_size=0.01,
    smoothing_sigma=0.0,
    normalize_by_unit_count=True,
):
    """
    Builds one observed population activity trace x_r(t) per retained region on a shared time grid.

    Returns:
        time_vector: [T]
        x_t: [T, R]
        signal_metadata: dict[str, dict]
    """
    bins = np.arange(t_start, t_stop + bin_size, bin_size)
    num_bins = len(bins) - 1
    spike_times_dict = session.spike_times

    time_vector = bins[:-1]
    region_names = list(region_unit_map.keys())
    x_columns = []
    signal_metadata = {}

    for region in region_names:
        unit_ids = region_unit_map[region]
        population_counts = np.zeros(num_bins, dtype=np.float64)

        for uid in unit_ids:
            spikes = spike_times_dict[uid]
            spikes = spikes[(spikes >= t_start) & (spikes <= t_stop)]
            counts, _ = np.histogram(spikes, bins=bins)
            population_counts += counts

        if normalize_by_unit_count:
            denom = max(len(unit_ids), 1) * bin_size
        else:
            denom = 1.0
        region_signal = population_counts / denom

        if smoothing_sigma > 0:
            sigma_bins = smoothing_sigma / bin_size
            region_signal = gaussian_filter1d(region_signal, sigma=sigma_bins)

        region_signal = region_signal.astype(np.float32)
        x_columns.append(region_signal)
        signal_metadata[region] = {
            "num_units": int(len(unit_ids)),
            "mean_activity": float(np.mean(region_signal)),
            "std_activity": float(np.std(region_signal)),
            "min_activity": float(np.min(region_signal)),
            "max_activity": float(np.max(region_signal)),
        }

    x_t = np.stack(x_columns, axis=-1).astype(np.float32)
    return time_vector, x_t, signal_metadata
