import json
import os
import requests
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

# Path to the JSON file
img_paths = read_json("/nfs/turbo/umms-drjieliu/proj/medlineKG/data/figure_json_by_article/pmcimage_paths_short.json")

def get_image_url(args):
    url, idx, directory = args
    try:
        responseImg = requests.get(url, timeout=10)
        if responseImg.status_code == 200:
            image_data = responseImg.content
            output_path = os.path.join(directory, f'{idx}.jpg')
            with open(output_path, 'wb') as file:
                file.write(image_data)
            return True
    except requests.exceptions.RequestException as e:
        # Handle errors (e.g., timeouts, connection errors)
        return False
    return False

def download_images_in_chunks(img_paths, chunk_size=1000, num_workers=None):
    if num_workers is None:
        num_workers = cpu_count()

    total_images = len(img_paths)
    num_chunks = (total_images + chunk_size - 1) // chunk_size  # Calculate the number of chunks

    for chunk_idx in range(num_chunks):
        # Create a directory for the current chunk
        chunk_dir = f'/scratch/drjieliu_root/drjieliu/panyijun/chunks/chunk_{chunk_idx}'
        os.makedirs(chunk_dir, exist_ok=True)

        # Determine the range of indices for the current chunk
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_images)

        # Prepare the arguments for multiprocessing
        args = [(img_paths[i], i, chunk_dir) for i in range(start_idx, end_idx)]

        # Use multiprocessing to download the images in the current chunk
        with Pool(num_workers) as pool:
            list(tqdm(pool.imap(get_image_url, args), total=len(args)))

def main():
    download_images_in_chunks(img_paths, chunk_size=200000)

if __name__ == "__main__":
    main()
