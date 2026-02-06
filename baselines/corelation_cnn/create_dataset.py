import pandas as pd
import numpy as np
import pickle
import torch
import cv2
import json
import os
import math
import collections
from tqdm import tqdm
from PIL import Image
from transformers import CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

# 1. Configuration & Paths
TARGET_METRIC = 'Recall@10'  # Options: 'Recall@10', 'Reciprocal_Rank'
LIMIT_TRAIN = 20000
LIMIT_VAL = 5000
LIMIT_TEST = 5000  # Added for consistency

VIDEO_FOLDER = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/MSRVTT_processed/train_videos"
TEST_VIDEO_FOLDER = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/MSRVTT_processed/test_videos"
VIDEO_EXTENSION = ".mp4"

# New Structured Paths
DATA_ROOT = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM"
PATHS = {
    "train": {
        "csv": os.path.join(DATA_ROOT, "metrics/msrvtt_train.csv"),
        "jsonl": os.path.join(DATA_ROOT, "top100/msrvtt_train.jsonl"),
        "out": os.path.join(DATA_ROOT, "corelation_cnn_datasets/msrvtt_train_rec.pickle")
    },
    "val": {
        "csv": os.path.join(DATA_ROOT, "metrics/msrvtt_val.csv"),
        "jsonl": os.path.join(DATA_ROOT, "top100/msrvtt_val.jsonl"),
        "out": os.path.join(DATA_ROOT, "corelation_cnn_datasets/msrvtt_val_rec.pickle")
    },
    "test": {
        "csv": os.path.join(DATA_ROOT, "metrics/msrvtt_test.csv"),
        "jsonl": os.path.join(DATA_ROOT, "top100/msrvtt_test.jsonl"),
        "out": os.path.join(DATA_ROOT, "corelation_cnn_datasets/msrvtt_test_rec.pickle")
    }
}

# 2. Parameters & Model Setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES_TO_SAMPLE = 12
MAX_VIDEOS_PER_QUERY = 25
MAX_CACHE_SIZE = 30000

PREPROCESSOR = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    lambda image: image.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

vision_model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE)
vision_model.eval()

video_cache = collections.OrderedDict()


# --- Helper Functions ---

def get_video_embedding(video_id):
    """Extracts embeddings with an LRU cache limit."""
    if video_id in video_cache:
        video_cache.move_to_end(video_id)
        return video_cache[video_id]

    path = os.path.join(VIDEO_FOLDER, f"{video_id}{VIDEO_EXTENSION}")
    if not os.path.exists(path):
        path = os.path.join(TEST_VIDEO_FOLDER, f"{video_id}{VIDEO_EXTENSION}")

    cap = cv2.VideoCapture(path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    indices = np.linspace(0, total_frames - 1, NUM_FRAMES_TO_SAMPLE, dtype=int)
    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(PREPROCESSOR(Image.fromarray(frame)))
    cap.release()

    if not frames: return None

    with torch.no_grad():
        pixel_values = torch.stack(frames).to(DEVICE)
        visual_outputs = vision_model(pixel_values).image_embeds
        visual_outputs = visual_outputs / visual_outputs.norm(dim=-1, keepdim=True)
        video_emb = torch.mean(visual_outputs, dim=0, keepdim=True)
        video_emb = (video_emb / video_emb.norm(dim=-1, keepdim=True)).cpu().numpy()

    video_cache[video_id] = video_emb
    if len(video_cache) > MAX_CACHE_SIZE:
        video_cache.popitem(last=False)
    return video_emb


def generate_corr_matrix(video_ids):
    embeddings = []
    for v_id in video_ids[:MAX_VIDEOS_PER_QUERY]:
        emb = get_video_embedding(v_id)
        if emb is not None:
            embeddings.append(emb)

    if len(embeddings) < 2:
        return np.zeros((512, 512), dtype=np.float32)

    matrix = np.vstack(embeddings)
    # Correlation coefficient matrix across the feature dimension (512)
    corr = np.corrcoef(matrix.T)
    return np.nan_to_num(corr).astype(np.float32)


def load_and_align(csv_path, jsonl_path, target_metric, limit=None):
    if not os.path.exists(csv_path) or not os.path.exists(jsonl_path):
        print(f"Warning: Skipping missing files: {csv_path} or {jsonl_path}")
        return []

    df = pd.read_csv(csv_path)
    preds_map = {}
    with open(jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Find the query key that isn't 'gt'
            query_text = [k for k in data.keys() if k != 'gt'][0]
            preds_map[query_text] = data[query_text]

    aligned_data = []
    for _, row in df.iterrows():
        q = row['Query']
        if q in preds_map:
            aligned_data.append({'score': row[target_metric], 'video_ids': preds_map[q]})

        if limit and len(aligned_data) >= limit:
            break

    return aligned_data


def process_and_save_sequentially(data_list, out_path, desc):
    """Processes queries with resume capability using append mode."""
    if not data_list:
        return

    start_index = 0
    if os.path.exists(out_path):
        with open(out_path, "rb") as f:
            while True:
                try:
                    pickle.load(f)
                    start_index += 1
                except EOFError:
                    break
        print(f"Resuming {desc} from index {start_index}/{len(data_list)}...")

    if start_index >= len(data_list):
        print(f"{desc} already completed.")
        return

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(out_path, "ab") as f:
        for item in tqdm(data_list[start_index:], desc=desc, initial=start_index, total=len(data_list)):
            corr = generate_corr_matrix(item['video_ids'])
            stacked_corr = np.expand_dims(corr, axis=0)  # [1, 512, 512] for CNN input
            pickle.dump((stacked_corr, item['score']), f)


# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print(f"Aligning files for metric: {TARGET_METRIC}...")

    # Load each split individually from their respective new paths
    train_data = load_and_align(PATHS["train"]["csv"], PATHS["train"]["jsonl"], TARGET_METRIC, limit=LIMIT_TRAIN)
    val_data = load_and_align(PATHS["val"]["csv"], PATHS["val"]["jsonl"], TARGET_METRIC, limit=LIMIT_VAL)
    test_data = load_and_align(PATHS["test"]["csv"], PATHS["test"]["jsonl"], TARGET_METRIC, limit=LIMIT_TEST)

    # Process each split
    process_and_save_sequentially(train_data, PATHS["train"]["out"], "Processing Train")
    process_and_save_sequentially(val_data, PATHS["val"]["out"], "Processing Val")
    process_and_save_sequentially(test_data, PATHS["test"]["out"], "Processing Test")

    print("\nAll datasets processed and saved successfully.")