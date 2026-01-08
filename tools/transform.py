import json
import cv2
from moviepy import VideoFileClip
import os
from pathlib import Path
import glob
import re


def get_video_fps(video_path):
    """Get the FPS (frames per second) of a video file."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    clip = VideoFileClip(video_path)
    fps = clip.fps
    clip.close()
    
    if fps == 0:
        raise ValueError(f"Unable to get video FPS: {video_path}")
    
    return fps


def time_to_frame(time_seconds, fps):
    """Convert time in seconds to frame number."""
    return int(round(time_seconds * fps))


def convert_single_json(input_json_path, video_path, output_json_path):
    """Convert a single JSON file with time-based breaks to frame-based sections."""
    with open(input_json_path, 'r', encoding='utf-8') as f:
        input_data = json.load(f)
    
    fps = get_video_fps(video_path)
    
    breaks = input_data.get('breaks', [])
    
    if not breaks:
        print(f"  Warning: No breaks data found")
        return None
    
    all_section_list = []
    selected_section_list = []
    
    start_frame = 0
    
    for break_point in breaks:
        time_seconds = break_point['time']
        end_frame = time_to_frame(time_seconds, fps)
        
        section = {
            "start": start_frame,
            "end": end_frame
        }
        
        all_section_list.append(section)
        selected_section_list.append(section)
        
        start_frame = end_frame
    
    output_data = {
        "all_section_list": all_section_list,
        "selected_section_list": selected_section_list,
        "duplicated_section_list": []
    }
    
    output_dir = os.path.dirname(output_json_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    return output_data, fps, len(all_section_list)


def extract_base_name(json_filename):
    """Extract base name from JSON filename by removing suffixes."""
    suffixes_to_remove = [
        '.mp4.breaks.json',
        '.avi.breaks.json', 
        '.mov.breaks.json',
        '.mkv.breaks.json',
        '.flv.breaks.json',
        '.wmv.breaks.json',
        '.webm.breaks.json',
        '.breaks.json',
        '.json'
    ]
    
    for suffix in suffixes_to_remove:
        if json_filename.endswith(suffix):
            return json_filename[:-len(suffix)]
    
    return json_filename


def find_matching_video(json_filename, video_folder, video_extensions=['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']):
    """Find the matching video file for a given JSON filename."""
    base_name = extract_base_name(json_filename)
    
    for ext in video_extensions:
        video_path = os.path.join(video_folder, base_name + ext)
        if os.path.exists(video_path):
            return video_path
    
    return None


def batch_convert_json(input_json_folder, video_folder, output_folder):
    """Batch convert JSON files from time-based to frame-based format."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Created output folder: {output_folder}")
    
    json_files = glob.glob(os.path.join(input_json_folder, "*.json"))
    
    if not json_files:
        print(f"Error: No JSON files found in {input_json_folder}")
        return
    
    print(f"Found {len(json_files)} JSON files")
    print("=" * 60)
    
    success_count = 0
    failed_count = 0
    failed_files = []
    
    for i, input_json_path in enumerate(json_files, 1):
        json_filename = os.path.basename(input_json_path)
        print(f"\n[{i}/{len(json_files)}] Processing: {json_filename}")
        
        try:
            video_path = find_matching_video(json_filename, video_folder)
            
            if video_path is None:
                base_name = extract_base_name(json_filename)
                print(f"  Extracted base name: {base_name}")
                print(f"  ❌ Error: Matching video file not found")
                print(f"  Expected video files: {base_name}.mp4, {base_name}.avi, etc.")
                failed_count += 1
                failed_files.append((json_filename, "Video file not found"))
                continue
            
            print(f"  Video file: {os.path.basename(video_path)}")
            
            output_json_path = os.path.join(output_folder, json_filename)
            
            result = convert_single_json(input_json_path, video_path, output_json_path)
            
            if result:
                output_data, fps, section_count = result
                print(f"  ✓ Success | FPS: {fps} | Sections: {section_count}")
                print(f"  Output: {output_json_path}")
                success_count += 1
            else:
                failed_count += 1
                failed_files.append((json_filename, "Conversion failed"))
                
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            failed_count += 1
            failed_files.append((json_filename, str(e)))
    
    print("\n" + "=" * 60)
    print("Processing completed!")
    print(f"Successful: {success_count} files")
    print(f"Failed: {failed_count} files")
    
    if failed_files:
        print("\nFailed files:")
        for filename, error in failed_files:
            print(f"  - {filename}: {error}")


if __name__ == "__main__":
    input_json_folder = "/path/to/input/json/folder"
    video_folder = "/path/to/video/folder"
    output_folder = "/path/to/output/folder"
    
    batch_convert_json(input_json_folder, video_folder, output_folder)