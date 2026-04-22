from datasets import load_dataset
import yt_dlp
import os
from tqdm import tqdm
import subprocess

# The other split is test_1k
ds = load_dataset("friedrichor/MSR-VTT", "train_9k")

video_dir = "MSRVTT_processed/train_videos"
audio_dir = "MSRVTT_processed/train_audios"
local_fallback_dir = "MSRVTT/videos/all"
os.makedirs(video_dir, exist_ok=True)
os.makedirs(audio_dir, exist_ok=True)

def split_video_audio(input_path, video_out, audio_out):
    temp_video = video_out + ".tmp.mp4"

    # Extract video only
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path, "-c:v", "copy", "-an", temp_video
    ], check=True)
    os.replace(temp_video, video_out)

    # Check if video has audio
    result = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index",
         "-of", "csv=p=0", input_path],
        capture_output=True, text=True
    )

    if result.stdout.strip():  # audio exists
        subprocess.run([
            "ffmpeg", "-y", "-i", input_path, "-vn", "-c:a", "mp3", "-b:a", "192k", audio_out
        ], check=True)
        print(f"Extracted audio: {audio_out}")
    else:
        print(f"No audio stream found in {input_path}, skipping audio extraction.")

for i, example in tqdm(enumerate(ds['train'])):
    url = example['url']
    output_video_path = os.path.join(video_dir, f"{example['video_id']}.mp4")
    output_audio_path = os.path.join(audio_dir, f"{example['video_id']}.mp3")

    fallback_path = os.path.join(local_fallback_dir, f"{example['video_id']}.mp4")

    split_video_audio(fallback_path, output_video_path, output_audio_path)

