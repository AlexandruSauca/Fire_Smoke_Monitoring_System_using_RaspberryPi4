import zipfile
import os

def unzip_file(zip_path, extract_to='.'):
    """
    Unzips a file to the specified directory.
    
    Args:
        zip_path (str): Path to the .zip file
        extract_to (str): Target directory (default: current dir)
    """
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        print(f"Unzipped to: {os.path.abspath(extract_to)}")
    except Exception as e:
        print(f"Error unzipping: {e}")

# Example usage
unzip_file('runs_best_no_aug.zip', extract_to='runs_best_no_aug')