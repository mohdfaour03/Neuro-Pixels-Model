import os
import requests
from tqdm import tqdm
from pathlib import Path

def download_file_resume(url, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    headers = {}
    file_size = 0
    if filepath.exists():
        file_size = filepath.stat().st_size
        headers['Range'] = f'bytes={file_size}-'
        print(f"File {filepath.name} partially exists. Resuming from {file_size / (1024**2):.2f} MB...")
        mode = 'ab'
    else:
        print(f"Starting fresh download for {filepath.name}...")
        mode = 'wb'
        
    try:
        # Increase the connection timeout significantly
        response = requests.get(url, headers=headers, stream=True, timeout=60)
        
        # 416 Requested Range Not Satisfiable typically means we already have the whole file
        if response.status_code == 416:
            head_res = requests.head(url)
            if head_res.status_code == 200:
                expected_size = int(head_res.headers.get('content-length', 0))
                if file_size >= expected_size:
                    print(f"File {filepath} is fully downloaded ({file_size / (1024**3):.2f} GB)!")
                    return
        
        # If server ignored our Range request, it will return 200 OK
        if response.status_code == 200 and file_size > 0:
            print("Server ignored Range header, starting from scratch...")
            mode = 'wb'
            file_size = 0
                
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0)) + file_size
        
        with open(filepath, mode) as f:
            with tqdm(total=total_size, initial=file_size, unit='B', unit_scale=True, desc=filepath.name) as pbar:
                # 1 MB chunks for speed
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
        print("\nDownload finished seamlessly!")
        
    except requests.exceptions.RequestException as e:
        print(f"\nDownload interrupted! Error: {e}")
        print("Please just run `python robust_download.py` again; it will pick up right where it stopped!")

if __name__ == "__main__":
    # We explicitly pull down the single huge NWB file for session 715093703
    url = "http://api.brain-map.org//api/v2/well_known_file_download/1026124469"
    target = "data/raw/session_715093703/session_715093703.nwb"
    download_file_resume(url, target)
