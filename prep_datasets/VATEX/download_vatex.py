import json
import os
import yt_dlp
import subprocess

from yt_dlp.utils import download_range_func

ANNOTATION_FILE = "VATEX/vatex_training_v1.0.json"
BASE_DIR = "VATEX/train"
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
AUDIO_DIR = os.path.join(BASE_DIR, "audios")

os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)


def download_clip(entry):
    id = entry["videoID"]
    youtube_id = id[:-14]
    _, start, end = id[-14:].split("_")
    clip_name = f"{entry['videoID']}.mp4"
    clip_path = os.path.join(VIDEO_DIR, clip_name)

    if os.path.exists(clip_path):
        return clip_path

    ydl_opts = {
        "format": "mp4[height<=480]",
        "outtmpl": clip_path,
        # yt-dlp reads the requested sections from `download_ranges`, and expects a
        # callable with the signature (info_dict, ydl). `download_sections` is the name
        # of the command line flag, not of the API option, and unknown keys are ignored
        # without any warning: the default is a single section with no start and no end,
        # which downloads the whole video.
        "download_ranges": download_range_func(None, [(int(start), int(end))]),
        # Without this the cut lands on the nearest keyframe, which can be seconds away
        # from the annotated segment.
        "force_keyframes_at_cuts": True,
        "cookiesfrombrowser": ("firefox",),
        "quiet": True
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={youtube_id}"])
        print(f"Saved video: {clip_name}")
        return clip_path
    except Exception as e:
        print(f"Failed {youtube_id}: {e}")
        return None


def split_audio_video(clip_path):
    if clip_path is None:
        return

    filename = os.path.splitext(os.path.basename(clip_path))[0]
    audio_path = os.path.join(AUDIO_DIR, f"{filename}.mp3")
    video_only_path = os.path.join(VIDEO_DIR, f"{filename}.mp4")

    # Extract audio
    subprocess.run([
        "ffmpeg", "-i", clip_path, "-q:a", "0", "-map", "a", audio_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Extract video without audio
    subprocess.run([
        "ffmpeg", "-i", clip_path, "-c", "copy", "-an", video_only_path, "-y"
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    print(f" -> Audio saved: {audio_path}")
    print(f" -> Video (no audio) saved: {video_only_path}")


def main():
    with open(ANNOTATION_FILE, "r") as f:
        data = json.load(f)

    for entry in data:
        if not os.path.exists(os.path.join(VIDEO_DIR, (entry["videoID"] + ".mp4"))):
            clip_path = download_clip(entry)
            split_audio_video(clip_path)


if __name__ == "__main__":
    main()
