import os
import json
from pathlib import Path
from typing import Dict, List

def find_all_json_files(root_folder: str) -> List[str]:
    """Recursively find all JSON files in the specified folder"""
    json_files = []
    for root, dirs, files in os.walk(root_folder):
        for file in files:
            if file.endswith('.json'):
                json_files.append(os.path.join(root, file))
    return json_files

def extract_scores(json_file_path: str) -> Dict[str, float]:
    """Extract scores from JSON file"""
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        blind_test_info = data.get('blind_test_info', {})
        
        scores = {
            'Narrative_Coherence': blind_test_info.get('Narrative_Coherence', {}).get('Score', None),
            'Visual_Fluency': blind_test_info.get('Visual_Fluency', {}).get('Score', None),
            'Pacing_Quality': blind_test_info.get('Pacing_Quality', {}).get('Score', None),
            'Content_Quality': blind_test_info.get('Content_Quality', {}).get('Score', None),
            'Overall_Appeal': blind_test_info.get('Overall_Appeal', {}).get('Score', None),
            'blind_test_avg_score': blind_test_info.get('blind_test_avg_score', None)
        }
        
        return scores
    except Exception as e:
        print(f"Error reading file {json_file_path}: {e}")
        return {}

def calculate_average_scores(root_folder: str):
    """Calculate average scores across all JSON files"""
    json_files = find_all_json_files(root_folder)
    
    if not json_files:
        print(f"No JSON files found in folder '{root_folder}'")
        return
    
    print(f"Found {len(json_files)} JSON files\n")
    
    # Initialize accumulators
    score_sums = {
        'Narrative_Coherence': 0,
        'Visual_Fluency': 0,
        'Pacing_Quality': 0,
        'Content_Quality': 0,
        'Overall_Appeal': 0,
        'blind_test_avg_score': 0
    }
    
    # Track the number of valid files for each metric
    score_counts = {key: 0 for key in score_sums.keys()}
    
    # Store scores for each file for detailed output
    all_file_scores = []
    
    # Iterate through all JSON files and accumulate scores
    for i, json_file in enumerate(json_files, 1):
        scores = extract_scores(json_file)
        if scores:
            print(f"[{i}/{len(json_files)}] Processing file: {os.path.basename(json_file)}")
            all_file_scores.append((json_file, scores))
            
            for key, value in scores.items():
                if value is not None:
                    score_sums[key] += value
                    score_counts[key] += 1
    
    print("\n" + "="*80)
    print("Statistical Results")
    print("="*80)
    
    # Calculate and output averages
    print(f"\nProcessed {len(all_file_scores)} valid JSON files in total\n")
    print("Average scores for each metric:")
    print("-" * 80)
    
    averages = {}
    for key in score_sums.keys():
        if score_counts[key] > 0:
            avg = score_sums[key] / score_counts[key]
            averages[key] = avg
            print(f"{key:25s}: {avg:.4f} (based on {score_counts[key]} files)")
        else:
            print(f"{key:25s}: No data available")
    
    print("\n" + "="*80)
    
    show_details = input("\nShow detailed scores for each file? (y/n): ").lower()
    if show_details == 'y':
        print("\n" + "="*80)
        print("Detailed Score List")
        print("="*80)
        for file_path, scores in all_file_scores:
            print(f"\nFile: {file_path}")
            for key, value in scores.items():
                print(f"  {key:25s}: {value if value is not None else 'N/A'}")
    
    return averages

if __name__ == "__main__":
    root_folder = "/path/to/evaluation/results"
    
    root_folder = root_folder.strip('"').strip("'")
    
    if not os.path.exists(root_folder):
        print(f"Error: Folder '{root_folder}' does not exist!")
    elif not os.path.isdir(root_folder):
        print(f"Error: '{root_folder}' is not a directory!")
    else:
        averages = calculate_average_scores(root_folder)