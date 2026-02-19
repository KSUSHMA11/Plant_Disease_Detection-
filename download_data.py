import gdown
import zipfile
import os

def download_and_extract(file_id, output_dir):
    url = f'https://drive.google.com/uc?id={file_id}'
    output_file = os.path.join(output_dir, 'dataset.zip')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Downloading dataset to {output_file}...")
    gdown.download(url, output_file, quiet=False)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(output_file, 'r') as zip_ref:
        zip_ref.extractall(output_dir)
        
    print("Done!")

if __name__ == "__main__":
    FILE_ID = '1J07k7Xzb--Vet7oqFG_Uk4AX6mwidCKu'
    OUTPUT_DIR = 'dataset'
    download_and_extract(FILE_ID, OUTPUT_DIR)
