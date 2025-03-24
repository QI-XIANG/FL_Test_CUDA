import os
import zipfile

def create_zip_of_result_dir(zip_filename="result_archive.zip", source_dir="./code/results"):
    """
    Creates a zip file of the ./code/result directory.

    Args:
        zip_filename (str): The name of the zip file to create.
        source_dir (str): The directory to archive (./code/result).
    """

    zip_path = os.path.join(".", zip_filename) #store the zip file in the current directory.

    if not os.path.exists(source_dir):
        print(f"Error: Directory '{source_dir}' does not exist.")
        return

    try:
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, _, files in os.walk(source_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, source_dir)
                    zipf.write(file_path, arcname)
        print(f"Zip file created successfully: {zip_path}")

    except Exception as e:
        print(f"Error creating zip file: {e}")

if __name__ == "__main__":
    create_zip_of_result_dir() #zip the result directory