import os
import gdown


def download_model_checkpoint(gdrive_file_id, output_path):
    """
    Download model checkpoint from Google Drive if it doesn't exist.

    Args:
        gdrive_file_id (str): Google Drive file ID
        output_path (str): Path where model will be saved
    """
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if file already exists
    if not os.path.exists(output_path):
        print(f"Downloading brain tumor model checkpoint to {output_path}...")
        url = f"https://drive.google.com/uc?id={gdrive_file_id}"
        gdown.download(url, output_path, quiet=False)
        print("Download complete!")
