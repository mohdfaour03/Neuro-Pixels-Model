import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

def extract_units(session, region_acronym="VISp", min_units=1):
    """
    Extracts high quality units from a specified brain region
    and specifically separates them into Excitatory and Inhibitory classes
    based on waveform_duration.
    """
    units = session.units
    # Filter by region
    region_units = units[units['ecephys_structure_acronym'] == region_acronym]
    
    # We rely on waveform_duration (> 0.4 ms -> Exc, <= 0.4 ms -> Inh)
    region_units = region_units.dropna(subset=['waveform_duration'])
    
    exc_units = region_units[region_units['waveform_duration'] > 0.4]
    inh_units = region_units[region_units['waveform_duration'] <= 0.4]
    
    exc_ids = exc_units.index.values
    inh_ids = inh_units.index.values
    
    print(f"[{region_acronym}] Found {len(exc_ids)} Excitatory units and {len(inh_ids)} Inhibitory units.")
    
    if len(exc_ids) + len(inh_ids) < min_units:
        raise ValueError(f"Not enough units found in {region_acronym}. Total found: {len(exc_ids)+len(inh_ids)}")
        
    return exc_ids, inh_ids

def build_population_activity(session, exc_ids, inh_ids, t_start, t_stop, bin_size=0.01, smoothing_sigma=0.0):
    """
    Builds a 2D population activity trace x(t) = [E(t), I(t)] from spike times.
    
    Args:
        session: AllenSDK session object
        exc_ids: array of Excitatory unit IDs
        inh_ids: array of Inhibitory unit IDs
        t_start: start time in seconds
        t_stop: stop time in seconds
        bin_size: bin duration in seconds
        smoothing_sigma: Gaussian filter std dev in seconds. If <= 0, no smoothing.
        
    Returns:
        time_bins (1D), x_t (2D array [N_bins, 2])
    """
    bins = np.arange(t_start, t_stop + bin_size, bin_size)
    num_bins = len(bins) - 1
    
    spike_times_dict = session.spike_times
    
    def get_population_rate(unit_ids):
        if len(unit_ids) == 0:
            return np.zeros(num_bins)
            
        population_counts = np.zeros(num_bins)
        for uid in unit_ids:
            spikes = spike_times_dict[uid]
            spikes = spikes[(spikes >= t_start) & (spikes <= t_stop)]
            counts, _ = np.histogram(spikes, bins=bins)
            population_counts += counts
            
        rate = population_counts / (len(unit_ids) * bin_size)
        if smoothing_sigma > 0:
            sigma_bins = smoothing_sigma / bin_size
            rate = gaussian_filter1d(rate, sigma=sigma_bins)
        return rate
        
    E_t = get_population_rate(exc_ids)
    I_t = get_population_rate(inh_ids)
    
    # Target shape: [N, 2]
    x_t = np.stack([E_t, I_t], axis=-1).astype(np.float32)
    time_vector = bins[:-1]
    
    return time_vector, x_t
