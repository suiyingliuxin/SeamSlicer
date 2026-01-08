import os
import pickle
import numpy as np
import torch
from utils import get_frames, get_batches
from visual import VisualModel
import ffmpeg
from moviepy import VideoFileClip
from audio import AliyunVideoASR
import re
import librosa
import json
from datetime import timedelta
import multiprocessing as mp


def load_model(checkpoint_path, device):
    print(f'Loading model from {checkpoint_path}')
    
    model = VisualModel().eval()
    
    if os.path.exists(checkpoint_path):
        model_dict = model.state_dict()
        pretrained_dict = torch.load(checkpoint_path, map_location=device)
        
        if 'net' in pretrained_dict:
            pretrained_dict = pretrained_dict['net']
        
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        print(f"Model has {len(model_dict)} parameters, loading {len(pretrained_dict)} parameters")
        
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    if device == "cuda":
        model = model.cuda()
    
    model.eval()
    return model

def predict_batch(model, batch, device):
    # Convert format: (H, W, C, T) -> (1, C, T, H, W)
    batch = torch.from_numpy(batch.transpose((3, 0, 1, 2))[np.newaxis, ...]).float()
    batch = batch.to(device)
    
    with torch.no_grad():
        output = model(batch)
        if isinstance(output, tuple):
            output = output[0]
        prediction = torch.sigmoid(output[0])
    
    return prediction.detach().cpu().numpy()

def predict_video(video_path, model, device, threshold=0.296):
    print(f"Processing video: {video_path}")
    
    frames = get_frames(video_path)
    total_frames = len(frames)
    print(f"Total frames: {total_frames}")
    
    predictions = []
    batch_count = 0
    
    for batch in get_batches(frames):
        batch_count += 1
        one_hot = predict_batch(model, batch, device)
        
        predictions.append(one_hot[25:75])
        
        if batch_count % 10 == 0:
            print(f"Processed {batch_count} batches...")
    
    predictions = np.concatenate(predictions, 0)[:total_frames]
    
    boundaries_binary = (predictions > threshold).astype(np.uint8)
    
    shot_boundaries = np.where(boundaries_binary == 1)[0].tolist()
    print(shot_boundaries)
    scenes = []
    if len(shot_boundaries) > 0:
        start = 0
        for boundary in shot_boundaries:
            if boundary > start:
                if boundary - start > 0:
                    scenes.append([start, boundary])
            start = boundary + 1
        if start < total_frames:
            if total_frames - start > 1:
                scenes.append([start, total_frames - 1])
    else:
        scenes.append([0, total_frames - 1])
    
    print(f"Detected {len(scenes)} shots")
    
    return predictions, shot_boundaries, scenes

def save_results(results, output_path):
    with open(output_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Results saved to {output_path}")

def split_video_by_scenes_with_audio(video_path, scenes, output_dir="output_scenes"):
    os.makedirs(output_dir, exist_ok=True)
    
    probe = ffmpeg.probe(video_path)
    video_info = next(s for s in probe['streams'] if s['codec_type'] == 'video')
    fps = eval(video_info['r_frame_rate'])
    
    print(f"Video info: FPS={fps}")
    print(f"Detected {len(scenes)} scene segments\n")
    
    for idx, (start_frame, end_frame) in enumerate(scenes):
        if end_frame < start_frame:
            print(f"Warning: Scene {idx} end frame ({end_frame}) < start frame ({start_frame}), corrected to same")
            end_frame = start_frame
        
        output_path = os.path.join(output_dir, f"scene_{idx:03d}_frame_{start_frame}-{end_frame}.mp4")
        
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        
        min_duration = float(1.0 / fps)-0.001  
        if duration < min_duration:
            duration = min_duration
            end_time = start_time + duration
            print(f"Warning: Scene {idx} duration too short, adjusted to {duration:.4f}s")
        
        try:
            if start_frame == end_frame:
                vf_filter = f"select='eq(n\\,{start_frame})',setpts=PTS-STARTPTS"
            else:
                vf_filter = f"select='between(n\\,{start_frame}\\,{end_frame-1})',setpts=PTS-STARTPTS"
            
            af_filter = f"atrim={start_time:.6f}:{end_time:.6f},asetpts=PTS-STARTPTS"
            
            (
                ffmpeg
                .input(video_path)
                .output(
                    output_path,
                    vf=vf_filter,
                    af=af_filter,
                    video_bitrate='5M',
                    audio_bitrate='192k'
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True)
            )
            
            print(f"✓ Scene {idx}: Frame {start_frame}-{end_frame} ({duration:.2f}s) -> {output_path}")
        
        except ffmpeg.Error as e:
            print(f"✗ Error: Failed to process scene {idx}")
            print(f"  Start frame: {start_frame}, End frame: {end_frame}")
            print(f"  Time range: {start_time:.4f}s - {end_time:.4f}s (duration: {duration:.4f}s)")
            print(f"  Error message: {e.stderr.decode()}")
            print()
    
    print(f"\nComplete! All segments saved to {output_dir} directory")

def get_video_fps(video_path):
    clip = VideoFileClip(video_path)
    fps = clip.fps
    clip.close()
    
    return fps

def has_intersection(interval1, interval2):
    start1, end1 = interval1
    start2, end2 = interval2
    return start1 <= end2 and start2 <= end1


def process_vision_scenes(vision_scenes, audio_frames):
    check_list=[]
    reserve_list=[]
    for vision_interval in vision_scenes:
        has_audio_intersection = False
        
        for audio_interval in audio_frames:
            if has_intersection(vision_interval, audio_interval):
                has_audio_intersection = True
                break
        
        if has_audio_intersection:
            # Add to review list
            check_list.append(vision_interval)
        else:
            # Directly reserve
            reserve_list.append(vision_interval)
    return check_list, reserve_list

def split_long_clip(video_path, final_list, model, device, results_cache, threshold=0.296, time_threshold=6,sentence_frame_list=[]):
    fps = get_video_fps(video_path)
    new_final_list = []
    
    for idx, (start_frame, end_frame) in enumerate(final_list):
        start_time = start_frame / fps
        end_time = end_frame / fps
        duration = end_time - start_time
        
        print(f"\nProcessing segment {idx}: Frame {start_frame}-{end_frame}, duration {duration:.2f}s")
        
        if duration <= time_threshold:
            new_final_list.append((start_frame, end_frame))
            print(f"✓ Duration appropriate, keep")
            continue
        
        print(f"Exceeds {time_threshold}s, performing speech recognition...")
        has_audio = False
        for sentence_interval in sentence_frame_list:
            if has_intersection((start_frame, end_frame), sentence_interval):
                has_audio = True
                break
        
        if has_audio:
            print("Word count exceeds 2, keep directly")
            new_final_list.append((start_frame, end_frame))
        else:
            print(f"Less text, use visual segmentation")
            sub_segments = visual_audio_split(
                video_path, 
                start_frame, 
                end_frame, 
                fps,
                model, 
                device, 
                threshold,
                time_threshold
            )
            new_final_list.extend(sub_segments)
     
    return new_final_list


def visual_audio_split(video_path, start_frame, end_frame, fps, model, device, threshold, time_threshold):
    sub_predictions, sub_boundaries, sub_scenes = predict_video_segment(
        video_path, 
        start_frame, 
        end_frame, 
        model, 
        device, 
        threshold
    )

    absolute_scenes = [(s[0] + start_frame, s[1] + start_frame) for s in sub_scenes]
    
    print(f"[Visual segmentation] Detected {len(absolute_scenes)} sub-segments")
    
    result = []
    for sub_start, sub_end in absolute_scenes:
        sub_duration = (sub_end - sub_start) / fps
        
        if sub_duration <= time_threshold:
            result.append((sub_start, sub_end))
            print(f"Sub-segment {sub_start}-{sub_end} ({sub_duration:.2f}s) appropriate")
        else:
            print(f"Sub-segment {sub_start}-{sub_end} ({sub_duration:.2f}s) still too long, use audio segmentation")

            audio_segments = audio_energy_split(
                video_path,
                sub_start,
                sub_end,
                fps,
                time_threshold
            )
            result.extend(audio_segments)
    
    return result


def audio_energy_split(video_path, start_frame, end_frame, fps, time_threshold):
    print(f"[Audio segmentation] Start...")

    start_time = start_frame / fps
    end_time = end_frame / fps
    
    clip = VideoFileClip(video_path)
    video_duration = clip.duration
    
    start_time = max(0, min(start_time, video_duration))
    end_time = max(start_time, min(end_time, video_duration))
    
    if end_time <= start_time:
        print(f"[Audio segmentation] Invalid time range: {start_time:.2f}s - {end_time:.2f}s")
        clip.close()
        return [(start_frame, end_frame)]

    audio_clip = clip.subclipped(start_time, end_time).audio
    
    if audio_clip is None:
        print(f"No audio, force even split")
        return force_split_evenly(start_frame, end_frame, fps, time_threshold)
    
    audio_array = audio_clip.to_soundarray(fps=22050)
    if len(audio_array.shape) == 2:
        audio_array = audio_array.mean(axis=1)  
    
    clip.close()
    
    frame_length = 2048
    hop_length = 512
    energy = librosa.feature.rms(y=audio_array, frame_length=frame_length, hop_length=hop_length)[0]
    
    # Detect energy drop points
    energy_drops = []
    threshold_ratio = 0.6  
    
    for i in range(1, len(energy) - 1):
        if energy[i] < energy[i-1] * threshold_ratio and energy[i] < energy[i+1]:
            time_sec = librosa.frames_to_time(i, sr=22050, hop_length=hop_length)
            energy_drops.append(time_sec)
    
    print(f"[Audio segmentation] Detected {len(energy_drops)} energy change points")
    
    if not energy_drops:
        print(f"No obvious changes, force even split")
        return force_split_evenly(start_frame, end_frame, fps, time_threshold)
    
    duration = end_time - start_time
    boundaries = [0] + energy_drops + [duration]
    
    result = []
    current_start = 0
    
    for i in range(1, len(boundaries)):
        segment_duration = boundaries[i] - current_start
        
        if segment_duration >= (time_threshold - 2) or i == len(boundaries) - 1:
            abs_start = int(start_frame + current_start * fps)
            abs_end = int(start_frame + boundaries[i] * fps) - 1
            result.append((abs_start, abs_end))
            print(f"Split: {abs_start}-{abs_end} ({boundaries[i] - current_start:.2f}s)")
            current_start = boundaries[i]
    
    return result


def sentence_split(content, segments, start_frame, end_frame, fps):
    print(f"[Sentence segmentation] Start...")
    
    sentence_pattern = r'[。！？.!?]+'
    sentences = re.split(sentence_pattern, content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    chinese_only_sentences = []
    for sentence in sentences:
        chinese_chars = re.findall(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', sentence)
        chinese_text = ''.join(chinese_chars)
        if chinese_text: 
            chinese_only_sentences.append(chinese_text)
    
    sentences = chinese_only_sentences
    
    print(f"[Sentence segmentation] Identified {len(sentences)} sentences")
    
    result = []

    if segments and len(segments) > 0:
        current_start_frame = start_frame
        current_text = ""
        
        for seg in segments:
            seg_text = seg.get('text', '').strip()
            current_text += seg_text
            
            if re.search(sentence_pattern, seg_text):
                seg_end_time = seg.get('end', 0)
                current_end_frame = round(start_frame + seg_end_time * fps)
                
                if current_end_frame > current_start_frame:
                    result.append((current_start_frame, current_end_frame))
                    print(f"Sentence segment: {current_start_frame}-{current_end_frame}")
                    current_start_frame = current_end_frame + 1
                    current_text = ""
        
        if current_start_frame < end_frame:
            result.append((current_start_frame, end_frame))
            print(f"Sentence segment: {current_start_frame}-{end_frame}")
    
    return result if result else [(start_frame, end_frame)]


def predict_video_segment(video_path, start_frame, end_frame, model, device, threshold):
    frames = get_frames(video_path)
    segment_frames = frames[start_frame:end_frame+1]
    
    predictions = []
    for batch in get_batches(segment_frames):
        one_hot = predict_batch(model, batch, device)
        predictions.append(one_hot[25:75])
    
    predictions = np.concatenate(predictions, 0)[:len(segment_frames)]
    
    boundaries_binary = (predictions > threshold).astype(np.uint8)
    shot_boundaries = np.where(boundaries_binary == 1)[0].tolist()
    
    scenes = []
    if len(shot_boundaries) > 0:
        start = 0
        for boundary in shot_boundaries:
            if boundary > start:
                scenes.append([start, boundary])
            start = boundary + 1
        if start < len(segment_frames):
            scenes.append([start, len(segment_frames) - 1])
    else:
        scenes.append([0, len(segment_frames) - 1])
    
    return predictions, shot_boundaries, scenes


def force_split_evenly(start_frame, end_frame, fps, time_threshold):
    duration = (end_frame - start_frame) / fps
    n_segments = int(np.ceil(duration / time_threshold))
    
    result = []
    frames_per_segment = (end_frame - start_frame) / n_segments
    
    for i in range(n_segments):
        seg_start = int(start_frame + i * frames_per_segment)
        seg_end = int(start_frame + (i + 1) * frames_per_segment) - 1
        if i == n_segments - 1:
            seg_end = end_frame 
        result.append((seg_start, seg_end))
    
    print(f"Force split into {n_segments} segments")
    return result


def format_time_label(seconds):
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    milliseconds = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{milliseconds:03d}"

def save_breaks_json(video_name, final_split_list, fps, output_dir="./output_json"):
    os.makedirs(output_dir, exist_ok=True)
    
    breaks = []
    for i, (start_frame, end_frame) in enumerate(final_split_list):
        if i < len(final_split_list) - 1:
            break_time = end_frame / fps
            time_label = format_time_label(break_time)
            
            breaks.append({
                "time": round(break_time, 6),
                "label": time_label,
                "type": "strong"
            })
    
    result = {
        "file": f"{video_name}.mp4",
        "breaks": breaks
    }
    
    json_path = os.path.join(output_dir, f"{video_name}.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"✓ JSON saved: {json_path}")
    print(f"  Detected {len(breaks)} split points")
    
    return json_path


def process_single_video(video_path, checkpoint_path, threshold, output_dir, process_id, access_key_id, access_key_secret, bucket_name, region_id):
    results_cache = {}
    
    try:
        print(f"\n[Process {process_id}] Start processing: {os.path.basename(video_path)}")
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"[Process {process_id}] Loading model...")
        model = load_model(checkpoint_path, device)

        asr_model = AliyunVideoASR(access_key_id, access_key_secret, region_id)

        print(f"[Process {process_id}] Performing visual scene detection...")
        predictions, shot_boundaries, vision_scenes = predict_video(
            video_path, model, device, threshold
        )

        print(f"[Process {process_id}] Performing audio detection...")
        video_name = os.path.splitext(os.path.basename(video_path))[0]

        detector_result = asr_model.recognize_video(
            video_path=video_path,
            bucket_name=bucket_name,
            use_vocal_separation=True
        )
        sentence_frame_list=[]
        if detector_result:
            clip=VideoFileClip(video_path)
            fps=clip.fps
            clip.close()
            frames = get_frames(video_path)
            total_frames = len(frames)
            for sentence in detector_result:
                start = sentence['start_time']
                end = sentence['end_time']
                start_frame = asr_model.time_to_frame(start, fps)
                end_frame = asr_model.time_to_frame(end, fps)
                sentence_frame_list.append((start_frame, end_frame))

            filtered_shot_boundaries = []
            for shot_frame in shot_boundaries:
                is_in_any_interval = any(
                    start <= shot_frame <= end 
                    for start, end in sentence_frame_list
                )
                if not is_in_any_interval:
                    filtered_shot_boundaries.append(shot_frame)
            for start, end in sentence_frame_list:
                filtered_shot_boundaries.append(end)
            filtered_shot_boundaries = sorted(filtered_shot_boundaries)
            scenes = []
            if len(filtered_shot_boundaries) > 0:
                start = 0
                for boundary in filtered_shot_boundaries:
                    if boundary > start:
                        if boundary - start > 0:
                            scenes.append([start, boundary])
                    start = boundary + 1
                if start < total_frames:
                    if total_frames - start > 1:
                        scenes.append([start, total_frames - 1])
            else:
                scenes.append([0, total_frames - 1])
        else:
            scenes = vision_scenes

        print(f"[Process {process_id}] Processing long segments...")
        final_split_list = split_long_clip(
            video_path=video_path,
            final_list=scenes,
            model=model,
            device=device,
            results_cache=results_cache,
            threshold=threshold,
            time_threshold=6,
            sentence_frame_list=sentence_frame_list
        )
        
        print(f"[Process {process_id}] Final split result: {final_split_list}")
        
        fps = get_video_fps(video_path)
        save_breaks_json(
            video_name=video_name,
            final_split_list=final_split_list,
            fps=fps,
            output_dir=output_dir
        )
        
        print(f"[Process {process_id}] ✓ Completed processing: {os.path.basename(video_path)}")
        return True
        
    except Exception as e:
        print(f"[Process {process_id}] ✗ Processing failed: {os.path.basename(video_path)}")
        print(f"[Process {process_id}] Error message: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def worker_process(video_list, checkpoint_path, threshold, output_dir, process_id, gpu_id, access_key_id, access_key_secret, bucket_name, region_id):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    print(f"\n{'='*60}")
    print(f"[Process {process_id}] Started, GPU {gpu_id}")
    print(f"[Process {process_id}] Assigned {len(video_list)} videos")
    print(f"{'='*60}\n")
    
    success_count = 0
    fail_count = 0
    
    for i, video_path in enumerate(video_list, 1):
        print(f"\n[Process {process_id}] Processing progress: {i}/{len(video_list)}")
        
        result = process_single_video(
            video_path=video_path,
            checkpoint_path=checkpoint_path,
            threshold=threshold,
            output_dir=output_dir,
            process_id=process_id,
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            bucket_name=bucket_name,
            region_id=region_id
        )
        
        if result:
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"[Process {process_id}] Completed all tasks")
    print(f"[Process {process_id}] Success: {success_count}, Failed: {fail_count}")
    print(f"{'='*60}\n")


def split_list(lst, n):
    k, m = divmod(len(lst), n)
    return [lst[i * k + min(i, m):(i + 1) * k + min(i + 1, m)] for i in range(n)]


def main():
    # ==================== Configuration Parameters ====================
    # 1. Video path
    VIDEO_DIR = "/path/to/your/video/directory"
    
    # 2. Visual model checkpoint path
    CHECKPOINT_PATH = "/path/to/your/model/checkpoint.pth"
    
    # 3. Visual model detection threshold
    THRESHOLD = 0.296
    
    # 4. Output directory
    OUTPUT_DIR = "/path/to/output/directory"

    # OSS configuration
    ACCESS_KEY_ID = "YOUR_ACCESS_KEY_ID"
    ACCESS_KEY_SECRET = "YOUR_ACCESS_KEY_SECRET"
    BUCKET_NAME = "YOUR_BUCKET_NAME"
    REGION_ID = "YOUR_REGION_ID"

    # 6. Multi-process configuration
    NUM_PROCESSES = 4  # Set number of processes, adjust based on GPU memory
    GPU_ID = 0  # GPU ID to use
    
    video_files = []
    
    if os.path.isfile(VIDEO_DIR):
        video_files = [VIDEO_DIR]
    elif os.path.isdir(VIDEO_DIR):
        video_files = [
            os.path.join(VIDEO_DIR, f) 
            for f in os.listdir(VIDEO_DIR) 
            if f.endswith('.mp4')
        ]
    else:
        raise ValueError(f"Invalid path: {VIDEO_DIR}")
    
    if not video_files:
        print("No video files found!")
        return
    
    print(f"\n{'='*60}")
    print(f"Found {len(video_files)} video files")
    print(f"Will use {NUM_PROCESSES} processes for parallel processing")
    print(f"Using GPU {GPU_ID}")
    print(f"{'='*60}\n")
    
    video_lists = split_list(video_files, NUM_PROCESSES)
    
    for i, vlist in enumerate(video_lists):
        print(f"Process {i}: Assigned {len(vlist)} videos")
    print()
    
    processes = []
    
    for i in range(NUM_PROCESSES):
        p = mp.Process(
            target=worker_process,
            args=(
                video_lists[i],
                CHECKPOINT_PATH,
                THRESHOLD,
                OUTPUT_DIR,
                i,
                GPU_ID,
                ACCESS_KEY_ID,
                ACCESS_KEY_SECRET,
                BUCKET_NAME,
                REGION_ID
            )
        )
        p.start()
        processes.append(p)
        print(f"✓ Process {i} started")
    
    print(f"\nAll processes started, waiting for completion...\n")
    
    for i, p in enumerate(processes):
        p.join()
        print(f"✓ Process {i} finished")
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()