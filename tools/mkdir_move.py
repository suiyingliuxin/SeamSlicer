import os
import shutil

# Specify the base directory containing JSON files
base_dir = "/path/to/section/data/folder"

os.chdir(base_dir)

# 1. Create video_001 to video_030 folders
print("Creating folders...")
for i in range(1, 31):
    folder_name = f"video_{i:03d}"  # Format as video_001, video_002, ...
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created folder: {folder_name}")
    else:
        print(f"Folder already exists: {folder_name}")

print("\n" + "="*50 + "\n")

# 2. Move JSON files to corresponding folders
print("Moving JSON files...")

all_files = [f for f in os.listdir('.') if f.endswith('.json')]

for i in range(1, 31):
    folder_name = f"video_{i:03d}"
    number_prefix = f"{i:03d}"  # 001, 002, ... 030
    
    # Find all JSON files starting with this number
    # Possible formats: 001.json or 001.mp4.breaks.json
    matching_files = [f for f in all_files if f.startswith(number_prefix)]
    
    if matching_files:
        for json_file in matching_files:
            destination = os.path.join(folder_name, json_file)
            
            shutil.move(json_file, destination)
            print(f"Moved: {json_file} -> {folder_name}/")
    else:
        print(f"No JSON files found starting with {number_prefix}")

print("\nTask completed!")