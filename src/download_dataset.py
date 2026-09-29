"""
EMOTIVA — Dataset Downloader
Downloads and extracts the RAVDESS Audio Speech dataset from Zenodo.
"""
import os
import sys
import zipfile
import urllib.request
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

ZENODO_URL = "https://zenodo.org/api/records/1188976/files/Audio_Speech_Actors_01-24.zip/content"

def download_and_extract_ravdess(target_dir=config.DATASET_PATH):
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "Audio_Speech_Actors_01-24.zip")
    
    print(f"Connecting to Zenodo to download RAVDESS dataset...")
    print(f"Target directory: {target_dir}")
    
    req = urllib.request.Request(ZENODO_URL, headers={'User-Agent': 'Mozilla/5.0'})
    
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60) as response, open(zip_path, 'wb') as out_file:
        total_size = int(response.info().get('Content-Length', 0))
        downloaded = 0
        chunk_size = 1024 * 512  # 512 KB
        
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)
            downloaded += len(chunk)
            if total_size > 0:
                pct = downloaded / total_size * 100
                mb = downloaded / (1024 * 1024)
                total_mb = total_size / (1024 * 1024)
                sys.stdout.write(f"\rDownloading: {mb:.1f}/{total_mb:.1f} MB ({pct:.1f}%)")
                sys.stdout.flush()
            else:
                sys.stdout.write(f"\rDownloading: {downloaded / (1024*1024):.1f} MB")
                sys.stdout.flush()
                
    print(f"\nDownload finished in {time.time() - t0:.1f}s. Extracting files...")
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_dir)
        
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    print(f"Extraction complete! RAVDESS dataset is ready at {target_dir}")

if __name__ == '__main__':
    download_and_extract_ravdess()
