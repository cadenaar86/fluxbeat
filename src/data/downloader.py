import os
import requests
import zipfile
import argparse
from tqdm import tqdm
from .. import config

FMA_URLS = {
    'fma_small': 'https://os.unil.cloud.switch.ch/fma/fma_small.zip',
    'fma_medium': 'https://os.unil.cloud.switch.ch/fma/fma_medium.zip',
    'fma_large': 'https://os.unil.cloud.switch.ch/fma/fma_large.zip',
    'fma_full': 'https://os.unil.cloud.switch.ch/fma/fma_full.zip',
    'metadata': 'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'
}

def download_file(url, dest_path):
    if os.path.exists(dest_path):
        print(f"File {dest_path} already exists. Skipping download.")
        return

    print(f"Downloading {url} to {dest_path}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024 # 1MB

    with open(dest_path, 'wb') as file, tqdm(
        desc=dest_path,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def unzip_file(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)

def main():
    parser = argparse.ArgumentParser(description='Download FMA dataset')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'medium', 'large', 'full'], help='Dataset size')
    parser.add_argument('--metadata', action='store_true', default=True, help='Download metadata')
    
    args = parser.parse_args()

    # Create raw directory
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.METADATA_DIR, exist_ok=True)

    # 1. Download Metadata (always needed)
    if args.metadata:
        meta_zip = os.path.join(config.DATA_DIR, 'fma_metadata.zip')
        download_file(FMA_URLS['metadata'], meta_zip)
        # Extract metadata to specific folder? FMA zip contains a folder 'fma_metadata'
        unzip_file(meta_zip, config.DATA_DIR)
        # Move contents if necessary or just config loader to point to it
        # The zip extracts 'fma_metadata/tracks.csv' etc. inside DATA_DIR
        # Config.METADATA_DIR points to DATA_DIR/metadata. 
        # We might need to move fma_metadata/* to metadata/ or just symlink.
        # Helper: check content
        extracted_folder = os.path.join(config.DATA_DIR, 'fma_metadata')
        if os.path.exists(extracted_folder):
            # Move files to METADATA_DIR
            import shutil
            for f in os.listdir(extracted_folder):
                shutil.move(os.path.join(extracted_folder, f), config.METADATA_DIR)
            os.rmdir(extracted_folder)
            print("Metadata configured.")

    # 2. Download Data
    dataset_key = f"fma_{args.size}"
    data_zip = os.path.join(config.RAW_DATA_DIR, f"{dataset_key}.zip")
    download_file(FMA_URLS[dataset_key], data_zip)
    unzip_file(data_zip, config.RAW_DATA_DIR)
    
    print("Download and extraction complete.")

if __name__ == "__main__":
    main()
