import argparse
import json
import os
import re
import sys

sys.path.pop(0)
sys.path.insert(0, '.')

import yaml
from tqdm import tqdm

from llms.gemini import gemini_caption
from llms.gpt import gptv_caption
from utils.image import short_side_resize
from utils.misc import get_path_info_list
from utils.video import read_frames


def extract_reason_and_score(attribute):
    match = re.search(r'^(.*)\((\d+(\.\d+)?)\)$', attribute)
    if match:
        reason = match.group(1).strip()
        score = float(match.group(2))
        return reason, score
    return "", 0.0


# 【新增函数】：合并多个批次的评分结果
def merge_batch_results(response_list):
    """
    合并多个批次的响应，计算平均分
    """
    all_scores = {
        'Narrative Coherence': [],
        'Visual Fluency': [],
        'Pacing Quality': [],
        'Content Quality': [],
        'Overall Appeal': []
    }
    all_reasons = {
        'Narrative Coherence': [],
        'Visual Fluency': [],
        'Pacing Quality': [],
        'Content Quality': [],
        'Overall Appeal': []
    }
    
    for response_str in response_list:
        response_str = response_str.strip()
        matches = re.findall(r'\[(.*?)\]: (.*? \(\d+\.\d+\))', response_str)
        attribute_all = {key: value for key, value in matches}
        
        for key in all_scores.keys():
            if key in attribute_all:
                reason, score = extract_reason_and_score(attribute_all[key])
                all_scores[key].append(score)
                all_reasons[key].append(reason)
    
    # 计算平均分和合并理由
    merged_results = {}
    for key in all_scores.keys():
        if all_scores[key]:
            avg_score = sum(all_scores[key]) / len(all_scores[key])
            combined_reason = "; ".join(all_reasons[key])
            merged_results[key] = (combined_reason, avg_score)
        else:
            merged_results[key] = ("", 0.0)
    
    return merged_results


def handle_one_path_info(path_info):

    video_path = path_info['video_path']
    blind_test_path = path_info['blind_test_path']
    sample_type = path_info["sample_type"] if "sample_type" in path_info else None
    sample_size = path_info["sample_size"] if "sample_size" in path_info else None

    config_path = path_info['config_path']
    tool_config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    sample_rate = tool_config['sample_rate']
    short_side = tool_config['short_side']
    model = tool_config['model']
    max_try_times = tool_config['max_try_count']
    
    batch_size = tool_config.get('batch_size', 40)

    frame_list, sample_id_list, video_info = read_frames(video_path, None, sample_rate, sample_type, sample_size)

    frame_list = [short_side_resize(x, short_side)[0] for x in frame_list]

    print(f"Total frames: {len(frame_list)}, batch_size: {batch_size}")

    blind_test_info = {}

    for i in range(max_try_times):
        try:
            if len(frame_list) > batch_size:
                print(f"Frame count exceeds batch_size, processing in batches...")
                all_responses = []
                
                for batch_idx in range(0, len(frame_list), batch_size):
                    batch_frames = frame_list[batch_idx:batch_idx+batch_size]
                    batch_end = min(batch_idx + batch_size, len(frame_list))
                    print(f"Processing batch {batch_idx//batch_size + 1}: frames {batch_idx} to {batch_end}")
                    
                    if 'gpt' in model:
                        print("Use gpt to eval.")
                        batch_response = gptv_caption(batch_frames, tool_config)
                    elif 'gemini' in model:
                        print("Use gemini to eval.")
                        batch_response = gemini_caption(batch_frames, tool_config)
                    
                    if batch_response != tool_config['empty_caption']:
                        all_responses.append(batch_response)
                        print(f"Batch {batch_idx//batch_size + 1} response:", batch_response)
                
                if all_responses:
                    merged_results = merge_batch_results(all_responses)
                    material_richness_reason, material_richness_score = merged_results['Narrative Coherence']
                    appeal_reason, appeal_score = merged_results['Visual Fluency']
                    content_of_exciting_segments_reason, content_of_exciting_segments_score = merged_results['Pacing Quality']
                    amount_of_waste_footage_reason, amount_of_waste_footage_score = merged_results['Content Quality']
                    overall_appeal_reason, overall_appeal_score = merged_results['Overall Appeal']
                else:
                    print(f'All batches returned empty captions, video_path: {video_path}')
                    break
            else:
                if 'gpt' in model:
                    print("Use gpt to eval.")
                    response_str = gptv_caption(frame_list, tool_config)

                elif 'gemini' in model:
                    print("Use gemini to eval.")
                    response_str = gemini_caption(frame_list, tool_config)

                if response_str == tool_config['empty_caption']:
                    print(f'empty caption, video_path: {video_path}')

                print(response_str)
                response_str = response_str.strip()
                matches = re.findall(r'\[(.*?)\]: (.*? \(\d+\.\d+\))', response_str)
                attribute_all = {key: value for key, value in matches}

                print("Fixed response_str:", attribute_all)

                material_richness_reason, material_richness_score = extract_reason_and_score(
                    attribute_all['Narrative Coherence'])
                appeal_reason, appeal_score = extract_reason_and_score(attribute_all['Visual Fluency'])
                content_of_exciting_segments_reason, content_of_exciting_segments_score = extract_reason_and_score(
                    attribute_all['Pacing Quality'])
                amount_of_waste_footage_reason, amount_of_waste_footage_score = extract_reason_and_score(
                    attribute_all['Content Quality'])
                overall_appeal_reason, overall_appeal_score = extract_reason_and_score(
                    attribute_all['Overall Appeal'])

            blind_test_avg_score = (
                material_richness_score +
                appeal_score +
                content_of_exciting_segments_score +
                amount_of_waste_footage_score +
                overall_appeal_score
            ) / 5

            print("blind_test_avg_score", blind_test_avg_score)

            blind_test_info = dict(
                Narrative_Coherence={"Reason": material_richness_reason, "Score": material_richness_score},
                Visual_Fluency={"Reason": appeal_reason, "Score": appeal_score},
                Pacing_Quality={"Reason": content_of_exciting_segments_reason,
                                "Score": content_of_exciting_segments_score},
                Content_Quality={"Reason": amount_of_waste_footage_reason,
                                 "Score": amount_of_waste_footage_score},
                Overall_Appeal={"Reason": overall_appeal_reason, "Score": overall_appeal_score},
                blind_test_avg_score=blind_test_avg_score,
                sample_id_list=sample_id_list
            )
            break

        except Exception as e:
            print(f"Exception at gpt_get_attribute: {e}", "GPT get wrong formate, try again!")

    blind_test_info_all = dict(
        blind_test_info=blind_test_info,
        video_info=video_info,
    )

    with open(blind_test_path, 'w') as f:
        json.dump(blind_test_info_all, f, indent=2)


def main(args):
    config_path = args.config_path
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    read_data_config = [
        dict(
            dir_name='video_dir',
            type='video',
            path_name='video_path',
        ),
        dict(
            dir_name='blind_test_dir',
            type='json',
            path_name='blind_test_path',
            is_output=True,
        )
    ]

    path_info_list = get_path_info_list(config, read_data_config)
    print(f'len(path_info_list): {len(path_info_list)}')

    for path_info in tqdm(path_info_list, ncols=80):
        print(f"\nprocess {path_info['video_path']}")
        handle_one_path_info(path_info)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str)
    args = parser.parse_args()
    main(args)