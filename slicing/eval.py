'''
Evaluation metrics for video segmentation
'''
import numpy as np
import json
import os
from pathlib import Path
from typing import List, Tuple, Dict, Union

def weighted_precision(pred_times: List[float], 
                        duration: float,
                        random_list,
                       gt_hard_times: List[float], 
                       gt_soft_times: List[float], 
                       tolerance: float = 0.2,
                       hard_weight: float = 0.7,
                       soft_weight: float = 0.3) -> Dict:
    pred_times = np.array(pred_times)
    gt_hard_times = np.array(gt_hard_times) if len(gt_hard_times) > 0 else np.array([])
    gt_soft_times = np.array(gt_soft_times) if len(gt_soft_times) > 0 else np.array([])
    
    if len(pred_times) == 0:
        return {
            'precision': 0.0,
            'matched_hard': 0,
            'matched_soft': 0,
            'false_positives': 0
        }
    
    matched_hard = 0
    matched_soft = 0
    false_positives = 0
    pass_count = 0

    for pred in pred_times:
        gap=0
        segment_idx = -1 

        if len(gt_hard_times) > 0 and any(abs(gt - pred) <= tolerance for gt in gt_hard_times):
            matched_hard += 1
        elif len(gt_soft_times) > 0 and any(abs(gt - pred) <= tolerance for gt in gt_soft_times):
            matched_soft += 1
        elif len(gt_hard_times) > 0 or len(gt_soft_times) > 0:
            all_gt = np.concatenate([gt_hard_times, gt_soft_times]) if len(gt_hard_times) > 0 and len(gt_soft_times) > 0 else (gt_hard_times if len(gt_hard_times) > 0 else gt_soft_times)
            all_gt = np.sort(all_gt)

            if len(all_gt) != len(random_list):
                raise ValueError(f"all_gt length ({len(all_gt)}) does not match random_list length ({len(random_list)})")
            
            left_gt = all_gt[all_gt < pred]
            right_gt = all_gt[all_gt > pred]
            
            gap_threshold = 6.0 
            if len(left_gt) == 0 and len(right_gt) > 0:
                segment_idx = 0
                nearest_right = right_gt[0]
                gap = nearest_right
            elif len(left_gt) > 0 and len(right_gt) > 0:
                segment_idx = len(left_gt)
                nearest_left = left_gt[-1]  
                nearest_right = right_gt[0]  
                gap = nearest_right - nearest_left
        
            if gap > gap_threshold and segment_idx != -1 and segment_idx < len(all_gt) and random_list[segment_idx]:
                pass_count += 1

        else:
            false_positives += 1
    
    weighted_matches = matched_hard + matched_soft + pass_count
    precision = weighted_matches / len(pred_times) 
    
    return {
        'precision': precision,
        'matched_hard': matched_hard,
        'matched_soft': matched_soft,
        'false_positives': false_positives
    }

def weighted_recall(pred_times: List[float], 
                    gt_hard_times: List[float], 
                    gt_soft_times: List[float], 
                    tolerance: float = 0.2,
                    hard_weight: float = 0.7,
                    soft_weight: float = 0.3) -> Dict:
    pred_times = np.array(pred_times)
    gt_hard_times = np.array(gt_hard_times) if len(gt_hard_times) > 0 else np.array([])
    gt_soft_times = np.array(gt_soft_times) if len(gt_soft_times) > 0 else np.array([])
    
    recalled_hard = 0
    recalled_soft = 0
    
    if len(gt_hard_times) > 0:
        for gt in gt_hard_times:
            if len(pred_times) > 0 and any(abs(gt - pred) <= tolerance for pred in pred_times):
                recalled_hard += 1
    
    if len(gt_soft_times) > 0:
        for gt in gt_soft_times:
            if len(pred_times) > 0 and any(abs(gt - pred) <= tolerance for pred in pred_times):
                recalled_soft += 1
    
    total_gt = len(gt_hard_times) + len(gt_soft_times)
    if total_gt == 0:
        return {
            'recall': 0.0,
            'hard_recall': 0.0,
            'soft_recall': 0.0,
            'recalled_hard': 0,
            'recalled_soft': 0
        }
    
    hard_recall = recalled_hard / len(gt_hard_times) if len(gt_hard_times) > 0 else 0
    soft_recall = recalled_soft / len(gt_soft_times) if len(gt_soft_times) > 0 else 0
    recall = hard_recall * hard_weight + soft_recall * soft_weight
    
    return {
        'recall': recall,
        'hard_recall': hard_recall,
        'soft_recall': soft_recall,
        'recalled_hard': recalled_hard,
        'recalled_soft': recalled_soft
    }


def weighted_f1_score(pred_times: List[float], 
                        duration: float,
                        random_list,
                      gt_hard_times: List[float], 
                      gt_soft_times: List[float], 
                      tolerance: float = 0.2,
                      precision_hard_weight: float = 0.7,
                      precision_soft_weight: float = 0.3,
                      recall_hard_weight: float = 0.7,
                      recall_soft_weight: float = 0.3) -> float:

    p = weighted_precision(pred_times, duration, random_list,gt_hard_times, gt_soft_times, 
                          tolerance, precision_hard_weight, precision_soft_weight)['precision']
    r = weighted_recall(pred_times, gt_hard_times, gt_soft_times, 
                       tolerance, recall_hard_weight, recall_soft_weight)['recall']
    
    return 2 * p * r / (p + r) if (p + r) > 0 else 0


def comprehensive_evaluation(pred_times: List[float], 
                            duration : float,
                            random_list,
                            gt_hard_times: List[float], 
                            gt_soft_times: List[float], 
                            tolerance: float = 0.2,
                            hard_weight: float = 0.7,
                            soft_weight: float = 0.3) -> Dict:
    precision_result = weighted_precision(pred_times, duration,random_list, gt_hard_times, gt_soft_times, tolerance,hard_weight,soft_weight)
    recall_result = weighted_recall(pred_times, gt_hard_times, gt_soft_times, tolerance,hard_weight,soft_weight)
    f1 = weighted_f1_score(pred_times, duration,random_list, gt_hard_times, gt_soft_times, tolerance,hard_weight,soft_weight,hard_weight,soft_weight)
    
    return {
        'weighted_precision': precision_result['precision'],
        'weighted_recall': recall_result['recall'],
        'weighted_f1': f1,
        'hard_recall': recall_result['hard_recall'],
        'soft_recall': recall_result['soft_recall'],
        'matched_hard': precision_result['matched_hard'],
        'matched_soft': precision_result['matched_soft'],
        'false_positives': precision_result['false_positives'],
        'total_pred': len(pred_times),
        'total_gt_hard': len(gt_hard_times),
        'total_gt_soft': len(gt_soft_times)
    }


def load_ground_truth(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    gt_hard_times = []
    gt_soft_times = []
    random_list = []
    duration = data.get('duration')
    
    for break_point in data.get('breaks', []):
        random= break_point.get('random')
        random_list.append(random)

        time = break_point.get('time')
        break_type = break_point.get('type', '').lower()
        
        if time is not None:
            if break_type == 'strong':
                gt_hard_times.append(time)
            elif break_type == 'weak':
                gt_soft_times.append(time)
    
    return gt_hard_times, gt_soft_times, duration, random_list


def load_predictions(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    pred_times = []
    
    for break_point in data.get('breaks', []):
        time = break_point.get('time')
        if time is not None:
            pred_times.append(time)
    
    return pred_times


def evaluate_from_json(gt_json_path: str, 
                       pred_json_path: str,
                       tolerance: float = 0.2,
                       hard_weight: float = 0.7,
                       soft_weight: float = 0.3) -> Dict:

    gt_hard_times, gt_soft_times, duration, random_list = load_ground_truth(gt_json_path)
    
    pred_times= load_predictions(pred_json_path)
    
    results = comprehensive_evaluation(
        pred_times, 
        duration, 
        random_list,
        gt_hard_times, 
        gt_soft_times, 
        tolerance=tolerance,
        hard_weight=hard_weight,
        soft_weight=soft_weight
    )
    
    return results

def find_matching_files(gt_dir: str, pred_dir: str) -> List[Tuple[str, str, str]]:
    gt_path = Path(gt_dir)
    pred_path = Path(pred_dir)
    
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth directory does not exist: {gt_dir}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")
    
    gt_files = {}
    for file in gt_path.glob("*.json"):
        video_id = file.name.split('.')[0] 
        gt_files[video_id] = str(file)
    
    matched_files = []
    for file in pred_path.glob("*.json"):
        video_id = file.name.split('.')[0]
        
        if video_id in gt_files:
            matched_files.append((video_id, gt_files[video_id], str(file)))
    
    return matched_files


def batch_evaluate(gt_dir: str, 
                   pred_dir: str,
                   tolerance: float = 0.2,
                   hard_weight: float = 0.7,
                   soft_weight: float = 0.3,
                   save_results: bool = True,
                   output_file: str = "evaluation_results.json") -> Dict:
    matched_files = find_matching_files(gt_dir, pred_dir)
    
    if not matched_files:
        print("No matching file pairs found!")
        print(f"Please check the file naming convention:")
        print(f"GT directory: {gt_dir}")
        print(f"Prediction directory: {pred_dir}")
        return {}
    
    print(f"Found {len(matched_files)} matching file pairs\n")
    print("="*80)
    
    all_results = {}
    summary_stats = {
        'total_files': len(matched_files),
        'avg_precision': 0,
        'avg_recall': 0,
        'avg_f1': 0,
        'avg_hard_recall': 0,
        'avg_soft_recall': 0
    }
    
    for i, (video_id, gt_path, pred_path) in enumerate(matched_files, 1):
        print(f"\n[{i}/{len(matched_files)}] Evaluating video: {video_id}")
        print(f"GT file: {Path(gt_path).name}")
        print(f"Prediction file: {Path(pred_path).name}")
        print("-" * 80)
        
        try:
            results = evaluate_from_json(
                gt_path, 
                pred_path,
                tolerance=tolerance,
                hard_weight=hard_weight,
                soft_weight=soft_weight
            )
            
            print(f"   Evaluation results for {video_id}:")
            print(f"Precision: {results['weighted_precision']:.4f}")
            print(f"Recall:    {results['weighted_recall']:.4f}")
            print(f"F1:        {results['weighted_f1']:.4f}")
            all_results[video_id] = results
            
            summary_stats['avg_precision'] += results['weighted_precision']
            summary_stats['avg_recall'] += results['weighted_recall']
            summary_stats['avg_f1'] += results['weighted_f1']
            summary_stats['avg_hard_recall'] += results['hard_recall']
            summary_stats['avg_soft_recall'] += results['soft_soft_recall']
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            all_results[video_id] = {'error': str(e)}
    
    # Calculate average values
    n_files = len(matched_files)
    summary_stats['avg_precision'] /= n_files
    summary_stats['avg_recall'] /= n_files
    summary_stats['avg_f1'] /= n_files
    summary_stats['avg_hard_recall'] /= n_files
    summary_stats['avg_soft_recall'] /= n_files
    
    print("\n" + "="*80)
    print("Overall Evaluation Summary")
    print("="*80)
    print(f"Total files:              {summary_stats['total_files']}")
    print(f"Average Precision: {summary_stats['avg_precision']:.4f}")
    print(f"Average Recall:    {summary_stats['avg_recall']:.4f}")
    print(f"Average F1:        {summary_stats['avg_f1']:.4f}")
    print(f"Average Hard Recall:        {summary_stats['avg_hard_recall']:.4f}")
    print(f"Average Soft Recall:        {summary_stats['avg_soft_recall']:.4f}")
    print("="*80)
    
    if save_results:
        output_data = {
            'summary': summary_stats,
            'individual_results': all_results,
            'parameters': {
                'tolerance': tolerance,
                'hard_weight': hard_weight,
                'soft_weight': soft_weight
            }
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\nEvaluation results saved to: {output_file}")
    
    return {
        'summary': summary_stats,
        'individual_results': all_results
    }


if __name__ == '__main__':
    # Set directory paths
    gt_dir = '/path/to/ground_truth'  # Ground truth directory
    pred_dir = "/path/to/predictions"  # Prediction results directory
    
    results = batch_evaluate(
        gt_dir=gt_dir,
        pred_dir=pred_dir,
        tolerance=0.2,
        hard_weight=1,
        soft_weight=0,
        save_results=True,
        output_file='evaluation_results.json'
    )