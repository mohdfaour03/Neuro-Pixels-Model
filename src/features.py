import numpy as np

def build_stimulus_signal(session, t_start, t_stop, bin_size=0.01, stimulus_class=''):
    """
    Builds a simple binary input signal u(t) aligned to time bins.
    u(t) = 1 when a visual stimulus is on screen, 0 otherwise.
    
    Args:
        session: AllenSDK session object
        t_start: start time in seconds
        t_stop: stop time in seconds
        bin_size: bin duration in seconds
        stimulus_class: string to filter stimulus presentations (e.g., 'drifting_gratings').
                        If empty, all stimulus presentations are considered.
                        
    Returns:
        time_vector, u_t
    """
    bins = np.arange(t_start, t_stop + bin_size, bin_size)
    time_vector = bins[:-1]
    u_t = np.zeros_like(time_vector)
    
    stim_table = session.stimulus_presentations
    
    if stimulus_class:
        stim_table = stim_table[stim_table['stimulus_name'] == stimulus_class]
        
    # filter to the time range
    stim_table = stim_table[(stim_table['start_time'] <= t_stop) & (stim_table['stop_time'] >= t_start)]
    
    # We want to set u_t = 1 for any bin that overlaps with a stimulus presentation
    # For a simple approach, we use searchsorted
    for _, row in stim_table.iterrows():
        onset = max(row['start_time'], t_start)
        offset = min(row['stop_time'], t_stop)
        
        start_idx = int((onset - t_start) / bin_size)
        stop_idx = int((offset - t_start) / bin_size)
        
        # safely bound indices
        start_idx = max(0, min(start_idx, len(u_t) - 1))
        stop_idx = max(start_idx, min(stop_idx, len(u_t) - 1))
        
        u_t[start_idx:stop_idx+1] = 1.0
        
    return time_vector, u_t
