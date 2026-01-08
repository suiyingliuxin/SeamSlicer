import json
import time
import os
import subprocess
from pathlib import Path
from aliyunsdkcore.client import AcsClient
from aliyunsdkcore.request import CommonRequest
from aliyunsdkcore.acs_exception.exceptions import ClientException, ServerException
import re
from moviepy import VideoFileClip
from http import HTTPStatus
from dashscope.audio.asr import Transcription
import dashscope
import requests
import oss2

class AliyunVideoASR:
    
    def __init__(self, access_key_id, access_key_secret, region_id, api_key=None):
        dashscope.base_http_api_url = 'https://dashscope.aliyuncs.com/api/v1'
        dashscope.api_key = api_key or os.getenv('DASHSCOPE_API_KEY', 'YOUR_API_KEY_HERE')
        
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.region_id = region_id
    
    def extract_audio_from_video(self, video_path, output_audio_path=None):
        try:
            if output_audio_path is None:
                video_path_obj = Path(video_path)
                output_audio_path = video_path_obj.parent / f"{video_path_obj.stem}.wav"
            
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-vn',
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-y',
                str(output_audio_path)
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"✓ Audio extraction successful: {output_audio_path}")
            return str(output_audio_path)
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Audio extraction failed: {e.stderr.decode()}")
            raise
        except FileNotFoundError:
            print("✗ ffmpeg not found")
            raise
    
    def upload_to_oss(self, file_path, bucket_name, access_key_id, access_key_secret, object_name=None):
        try:
            
            auth = oss2.Auth(access_key_id, access_key_secret)
            bucket = oss2.Bucket(auth, f'https://oss-{self.region_id}.aliyuncs.com', bucket_name)
            
            if object_name is None:
                object_name = Path(file_path).name
            
            with open(file_path, 'rb') as f:
                bucket.put_object(object_name, f)
            
            file_url = f"https://{bucket_name}.oss-{self.region_id}.aliyuncs.com/{object_name}"
            print(f"✓ File upload successful: {file_url}")
            return file_url
            
        except ImportError:
            print("✗ Please install OSS SDK first: pip install oss2")
            raise
        except Exception as e:
            print(f"✗ File upload failed: {e}")
            raise
    
    def submit_task(self, file_url):
        try:
            task_response = Transcription.async_call(
                model='fun-asr',
                file_urls=[file_url]
            )
            
            if task_response.status_code == HTTPStatus.OK:
                task_id = task_response.output.task_id
                print(f"✓ Task submitted successfully, task ID: {task_id}")
                return task_id
            else:
                print(f"✗ Task submission failed: {task_response}")
                return None
                
        except (ServerException, ClientException) as e:
            print(f"✗ Task submission failed: {e}")
            raise
    
    def get_task_result(self, task_id, poll_interval=10):
        try:
            transcribe_response = Transcription.wait(task=task_id)
            if transcribe_response.status_code == HTTPStatus.OK:
                transcribe_output = transcribe_response.output
                status = transcribe_output['task_status']
            
                if status == "SUCCEEDED":
                    print(f"✓ Recognition completed!")
                    transcribe_url = transcribe_output['results'][0]['transcription_url']
                    response_data = requests.get(transcribe_url)
                    return response_data
                elif status in ["FAILED"]:
                    print(f"✗ Task failed: {status}")
                    raise Exception(f"Task failed: {status}")
                    return None
                elif status in ["PENDING", "RUNNING"]:
                    print(f"⏳ Task status: {status}, waiting...")
                    time.sleep(poll_interval)
                else:
                    print(f"⏳ Task status: {status}")
                    time.sleep(poll_interval)
            else:
                raise Exception(f"Task failed, transcribe_response.status_code: {transcribe_response.status_code}")
                
        except (ServerException, ClientException) as e:
            print(f"✗ Query failed: {e}")
            raise
        
    
    def parse_result(self, result_data):
        sentences = []
        
        if not result_data:
            return sentences
        
        result_json = result_data["transcripts"][0]
        for idx, sentence in enumerate(result_json["sentences"], 1):
            text = sentence.get('text', '')
            if not re.search(r'[\u4e00-\u9fff]', text):
                continue

            sentence_info = {
                'id': idx,
                'text': text,
                'start_time': sentence.get('begin_time', 0) / 1000,
                'end_time': sentence.get('end_time', 0) / 1000,
                'sentence_id': sentence.get('sentence_id', 0),
                'words': []
            }
            
            for word in sentence["words"]:
                word_info = {
                    'text': word.get('text', ''),
                    'start_time': word.get('begin_time', 0) / 1000,
                    'end_time': word.get('end_time', 0) / 1000
                }
                sentence_info['words'].append(word_info)
            
            sentences.append(sentence_info)
        
        return sentences
    

    def separate_vocals_cli(self, audio_path):
        output_dir = os.path.join(os.path.dirname(audio_path), 'demucs_output')
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            'demucs',
            '-n', 'htdemucs',
            '--two-stems', 'vocals',
            '-o', output_dir,
            audio_path
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        audio_name = os.path.splitext(os.path.basename(audio_path))[0]
        vocals_path = os.path.join(output_dir, 'htdemucs', audio_name, 'vocals.wav')
        
        final_path = audio_path
        cmd = [
            'ffmpeg',
            '-i', vocals_path,
            '-ar', '16000',
            '-ac', '1',
            '-y',
            final_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        import shutil
        shutil.rmtree(output_dir)
        
        return final_path
    

    def recognize_video(self, video_path, bucket_name, use_vocal_separation=False):
        print(f"\n{'='*50}")
        print(f"Starting video processing: {video_path}")
        print(f"{'='*50}\n")
        
        print("Step 1/4: Extracting audio")
        audio_path = self.extract_audio_from_video(video_path)
        if use_vocal_separation:
            print("Using vocal separation")
            audio_path = self.separate_vocals_cli(audio_path)

        print("\nStep 2/4: Uploading file to OSS")
        file_url = self.upload_to_oss(audio_path, bucket_name, self.access_key_id, self.access_key_secret)

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print("\nStep 3/4: Submitting recognition task")
        task_id = self.submit_task(file_url)
        
        if task_id is None:
            return None
        
        print("\nStep 4/4: Waiting for recognition results")
        result_data = self.get_task_result(task_id)
        
        if result_data is None:
            return None
        
        sentences = self.parse_result(result_data)
        
        return sentences
    
    def print_results(self, sentences, max_display=100):
        print("\nRecognition results preview:\n")
        print(len(sentences))
        
        for sentence in sentences[:max_display]:
            start = sentence['start_time']
            end = sentence['end_time']
            text = sentence['text']
            print(f"[{start:.2f}s - {end:.2f}s] {text}")
        
        if len(sentences) > max_display:
            print(f"\n... {len(sentences) - max_display} more sentences not displayed")

    def time_to_frame(self, time_seconds, fps):
        return int(round(time_seconds * fps))