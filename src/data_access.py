import os
from pathlib import Path
from allensdk.brain_observatory.ecephys.ecephys_project_cache import EcephysProjectCache

def setup_cache(data_dir: str):
    """
    Initializes the AllenSDK EcephysProjectCache at the specified directory.
    If the directory does not exist, it will be created.
    """
    data_dir_path = Path(data_dir)
    data_dir_path.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir_path / "manifest.json"
    cache = EcephysProjectCache.from_warehouse(manifest=manifest_path, timeout=100000)
    return cache

def get_session(session_id: int, data_dir: str):
    """
    Loads one specific Allen Brain Observatory Visual Coding Neuropixels session.
    Downloads the data if it isn't cached locally.
    
    Args:
        session_id (int): The session ID.
        data_dir (str): The local cache directory.
        
    Returns:
        The session object containing spike times, units, and stimulus presentations.
    """
    print(f"Loading session {session_id} from Allen Brain Observatory...")
    cache = setup_cache(data_dir)
    
    # Optional: fetch a list of sessions first to ensure valid ID
    sessions = cache.get_session_table()
    if session_id not in sessions.index:
        raise ValueError(f"Session {session_id} not found in the Ecephys dataset.")
        
    session = cache.get_session_data(session_id)
    print(f"Successfully loaded session {session_id}.")
    return session
