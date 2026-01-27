import torch
import cv2  # OpenCV
import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from PIL import Image
import math

# --- NEW IMPORTS based on the documentation ---
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

# -----------------------------------------------------------------
# ### --- CONFIGURATION (!!! YOU MUST EDIT THIS SECTION !!!) --- ###
# -----------------------------------------------------------------

# 1. Path to your TRAIN/VAL video folder
VIDEO_FOLDER = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/VATEX/train/videos/"  # e.g., /data/videos/train/

# 2. Path to your TEST video folder
TEST_VIDEO_FOLDER = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/VATEX/val/videos"  # <-- ADDED THIS

# 3. File extension of your videos
VIDEO_EXTENSION = ".mp4"

# 4. Path to your SINGLE input JSONL file (for train/val)
JSONL_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/top100/vatex_train.jsonl"

# 5. Paths for the OUTPUT .pickle files
TRAIN_OUTPUT_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets/vatex_train.pickle"
VAL_OUTPUT_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets/vatex_val.pickle"
TEST_OUTPUT_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets/vatex_test.pickle"

# 6. Path to your TEST JSONL file
TEST_JSONL_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/top100/vatex_val.jsonl"

# 7. Define your split. 0.2 means 20% for validation, 80% for train
VALIDATION_SPLIT_RATIO = 0.2

# 8. Model to use from Hugging Face
MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"

# 9. How many frames to sample from each video
NUM_FRAMES_TO_SAMPLE = 12

# 10. How many of the top-100 results to process
MAX_VIDEOS_PER_QUERY = 25

# -----------------------------------------------------------------
# ### --- END CONFIGURATION --- ###
# -----------------------------------------------------------------

# Setup device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# A cache to store video embeddings to avoid re-calculating
video_embedding_cache = {}

# --- NEW: Video Preprocessor from the HF Example ---
VIDEO_PREPROCESSOR = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    lambda image: image.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])


def load_video_frames(video_path, num_frames_to_sample, training=True):
    if not os.path.exists(video_path):
        print(f"Warning: Video file not found {video_path}")
        return torch.zeros(num_frames_to_sample, 3, 224, 224)  # Return padding

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or total_frames <= 0:
        # Return zero-tensor to prevent batch crash
        return torch.zeros(num_frames_to_sample, 3, 224, 224)

    # --- Correct CLIP4CLIP Sampling Strategy ---
    # Divide video into N equal segments
    seg_size = float(total_frames - 1) / num_frames_to_sample
    indices = []

    for i in range(num_frames_to_sample):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))

        if training:
            # Random sample in segment (Augmentation)
            idx = np.random.randint(start, end + 1)
        else:
            # Center sample (Deterministic for testing)
            idx = (start + end) // 2
        indices.append(idx)

    frame_tensors = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()

        # Fallback: if read fails, try to read the 0th frame or previous frame
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()

        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tensor = VIDEO_PREPROCESSOR(pil_image)
            frame_tensors.append(tensor)
        else:
            # Last resort: append a black frame to maintain shape
            frame_tensors.append(torch.zeros(3, 224, 224))

    cap.release()

    # Stack into a (num_frames, 3, 224, 224) tensor
    return torch.stack(frame_tensors).to(DEVICE)


def process_lines(lines, text_model, tokenizer, vision_model, pbar_desc="Processing"):
    """
    Helper function to process a list of JSONL lines.
    *** MODIFIED to check BOTH video folders (VIDEO_FOLDER and TEST_VIDEO_FOLDER) ***
    """
    dataset_pairs = []

    for line in tqdm(lines, desc=pbar_desc):
        try:
            data = json.loads(line)

            gt_id = data.get('gt')
            if gt_id is None or not isinstance(gt_id, str):
                print(f"Skipping line, missing 'gt' key or 'gt' is not a string: {line}")
                continue

            query_text, retrieved_ids = None, None
            for key, value in data.items():
                if key != 'gt':
                    query_text, retrieved_ids = key, value
                    break

            if query_text is None: continue

        except Exception as e:
            print(f"Skipping line, error parsing JSON: {e} | Line: {line}")
            continue

        with torch.no_grad():
            # 1. Get text embedding
            inputs = tokenizer(text=[query_text],
                               return_tensors="pt",
                               max_length=77,           # <-- ADDED: Set max length
                               padding='max_length',    # <-- ADDED: Pad to max length
                               truncation=True).to(DEVICE)
            outputs = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_emb_tensor = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
            text_emb = text_emb_tensor.detach().cpu().numpy()

            # 2. Process the top N retrieved videos
            for video_id in retrieved_ids[:MAX_VIDEOS_PER_QUERY]:

                video_emb = None

                if video_id in video_embedding_cache:
                    video_emb = video_embedding_cache[video_id]
                else:
                    # *** MODIFIED LOGIC: Check both folders ***

                    # 1. Check train folder first
                    train_video_path = os.path.join(VIDEO_FOLDER, f"{video_id}{VIDEO_EXTENSION}")
                    frame_tensor = load_video_frames(train_video_path, NUM_FRAMES_TO_SAMPLE)

                    # 2. If not found in train, check test folder
                    if frame_tensor is None:
                        test_video_path = os.path.join(TEST_VIDEO_FOLDER, f"{video_id}{VIDEO_EXTENSION}")
                        frame_tensor = load_video_frames(test_video_path, NUM_FRAMES_TO_SAMPLE)

                    # 3. If still not found, print warning and skip video
                    if frame_tensor is None:
                        print(f"Warning: Video {video_id} not found in {VIDEO_FOLDER} or {TEST_VIDEO_FOLDER}")
                        continue

                    frame_tensor = frame_tensor.to(DEVICE)
                        # --- Found video, now process it ---
                    visual_output = vision_model(frame_tensor)["image_embeds"]
                    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
                    visual_output = torch.mean(visual_output, dim=0)
                    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

                    video_emb = visual_output.unsqueeze(0).detach().cpu().numpy()
                    video_embedding_cache[video_id] = video_emb

                # 3. Assign score
                score = 1 if video_id == gt_id else 0

                # 4. Stack features
                combined_features = np.hstack((text_emb, video_emb))

                # 5. Append to dataset
                dataset_pairs.append((combined_features, score))

    return dataset_pairs


def generate_splits_from_file(text_model, tokenizer, vision_model):
    """
    Reads one file, splits it, and generates two datasets (train/val).
    """
    print(f"Reading all lines from {JSONL_PATH}...")
    try:
        with open(JSONL_PATH, 'r') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: Train/Val file not found at {JSONL_PATH}")
        print("Skipping train/val set generation.")
        return

    print(f"Total lines read: {len(all_lines)}")

    split_point = math.floor(len(all_lines) * (1.0 - VALIDATION_SPLIT_RATIO))
    train_lines, val_lines = all_lines[:split_point], all_lines[split_point:]

    print(f"Training queries: {len(train_lines)}")
    print(f"Validation queries: {len(val_lines)}")
    print("-" * 30)

    # --- 1. Process Training Data ---
    print("Processing Training Data...")
    train_dataset = process_lines(train_lines, text_model, tokenizer, vision_model, pbar_desc="Train")

    print(f"\nProcessed {len(train_dataset)} training pairs.")
    print(f"Saving training dataset to {TRAIN_OUTPUT_PATH}...")
    with open(TRAIN_OUTPUT_PATH, "wb") as f_out:
        pickle.dump(train_dataset, f_out)
    print("Training dataset saved.")
    print("-" * 30)

    # --- 2. Process Validation Data ---
    print("Processing Validation Data...")
    val_dataset = process_lines(val_lines, text_model, tokenizer, vision_model, pbar_desc="Validation")

    print(f"\nProcessed {len(val_dataset)} validation pairs.")
    print(f"Saving validation dataset to {VAL_OUTPUT_PATH}...")
    with open(VAL_OUTPUT_PATH, "wb") as f_out:
        pickle.dump(val_dataset, f_out)
    print("Validation dataset saved.")


def generate_test_dataset(text_model, tokenizer, vision_model):
    """
    Processes a separate test file.
    """
    print("-" * 30)
    print("Processing Test Data...")
    print(f"Reading all lines from {TEST_JSONL_PATH}...")

    try:
        with open(TEST_JSONL_PATH, 'r') as f:
            test_lines = f.readlines()
    except FileNotFoundError:
        print(f"ERROR: Test file not found at {TEST_JSONL_PATH}")
        print("Skipping test set generation.")
        return

    print(f"Total test queries: {len(test_lines)}")

    test_dataset = process_lines(test_lines, text_model, tokenizer, vision_model, pbar_desc="Test")

    print(f"\nProcessed {len(test_dataset)} test pairs.")
    print(f"Saving test dataset to {TEST_OUTPUT_PATH}...")
    with open(TEST_OUTPUT_PATH, "wb") as f_out:
        pickle.dump(test_dataset, f_out)
    print("Test dataset saved.")


if __name__ == "__main__":
    # 1. Load models and tokenizer ONCE
    print(f"Loading tokenizer and models from: {MODEL_NAME}")

    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)

    text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)

    text_model.eval()
    vision_model.eval()
    print("-" * 30)

    # 2. Generate Train/Val splits
    generate_splits_from_file(text_model, tokenizer, vision_model)

    # 3. Generate Test set
    generate_test_dataset(text_model, tokenizer, vision_model)

    print("\nAll processing complete.")