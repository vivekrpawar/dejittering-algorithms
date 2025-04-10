import os
import sys
def rename_files(directory):
    # Get a list of all files in the directory
    files = sorted([f for f in os.listdir(directory) if f.endswith(".png")])
    
    for idx, file in enumerate(files, start=1):
        # Generate new filename with 6-digit index
        index = int(file.split("_")[-1].split(".")[0])
        new_name = f"corrected_{index:06d}.png"
        
        # Get full paths
        old_path = os.path.join(directory, file)
        new_path = os.path.join(directory, new_name)
        
        # Rename file
        os.rename(old_path, new_path)
        print(f"Renamed: {file} -> {new_name}")

# Specify the directory where the files are located
directory_path = sys.argv[1]
rename_files(directory_path)