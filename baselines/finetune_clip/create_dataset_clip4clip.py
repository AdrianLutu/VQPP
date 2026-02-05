import torch
import cv2  # OpenCV
import numpy as np
import pickle
import json
import os
from tqdm import tqdm
from PIL import Image
import math
from transformers import CLIPTokenizer, CLIPTextModelWithProjection, CLIPVisionModelWithProjection
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, InterpolationMode

VIDEO_FOLDER = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/VATEX/train/videos/"
TEST_VIDEO_FOLDER = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/VATEX/val/videos"
VIDEO_EXTENSION = ".mp4"

TRAIN_JSONL_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/top100/vatex_train.jsonl"
VAL_JSONL_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/top100/vatex_val.jsonl"
TEST_JSONL_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/top100/vatex_test.jsonl"

TRAIN_OUTPUT_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets/vatex_train.pickle"
VAL_OUTPUT_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets/vatex_val.pickle"
TEST_OUTPUT_PATH = "/home/eduard/Desktop/Research/Adrian/VQPP/GRAM/GRAM/clip_datasets/vatex_test.pickle"

MODEL_NAME = "Searchium-ai/clip4clip-webvid150k"
NUM_FRAMES_TO_SAMPLE = 12
MAX_VIDEOS_PER_QUERY = 25

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

video_embedding_cache = {}

VIDEO_PREPROCESSOR = Compose([
    Resize(224, interpolation=InterpolationMode.BICUBIC),
    CenterCrop(224),
    lambda image: image.convert("RGB"),
    ToTensor(),
    Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def load_video_frames(video_path, num_frames_to_sample, training=True):
    if not os.path.exists(video_path):
        return None

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if not cap.isOpened() or total_frames <= 0:
        return None

    seg_size = float(total_frames - 1) / num_frames_to_sample
    indices = []

    for i in range(num_frames_to_sample):
        start = int(np.round(seg_size * i))
        end = int(np.round(seg_size * (i + 1)))
        if training:
            idx = np.random.randint(start, end + 1)
        else:
            idx = (start + end) // 2
        indices.append(idx)

    frame_tensors = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            tensor = VIDEO_PREPROCESSOR(pil_image)
            frame_tensors.append(tensor)
        else:
            frame_tensors.append(torch.zeros(3, 224, 224))

    cap.release()
    return torch.stack(frame_tensors).to(DEVICE)

def process_lines(lines, text_model, tokenizer, vision_model, pbar_desc="Processing"):
    dataset_pairs = []

    for line in tqdm(lines, desc=pbar_desc):
        try:
            data = json.loads(line)
            gt_id = data.get('gt')
            if gt_id is None or not isinstance(gt_id, str):
                continue

            query_text, retrieved_ids = None, None
            for key, value in data.items():
                if key != 'gt':
                    query_text, retrieved_ids = key, value
                    break
            if query_text is None: continue

        except Exception as e:
            print(f"Error parsing JSON: {e}")
            continue

        with torch.no_grad():
            inputs = tokenizer(text=[query_text],
                               return_tensors="pt",
                               max_length=77,           
                               padding='max_length', 
                               truncation=True).to(DEVICE)
            outputs = text_model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
            text_emb_tensor = outputs[0] / outputs[0].norm(dim=-1, keepdim=True)
            text_emb = text_emb_tensor.detach().cpu().numpy()

            for video_id in retrieved_ids[:MAX_VIDEOS_PER_QUERY]:
                video_emb = None

                if video_id in video_embedding_cache:
                    video_emb = video_embedding_cache[video_id]
                else:
                    video_path = os.path.join(VIDEO_FOLDER, f"{video_id}{VIDEO_EXTENSION}")
                    frame_tensor = load_video_frames(video_path, NUM_FRAMES_TO_SAMPLE)

                    if frame_tensor is None:
                        video_path = os.path.join(TEST_VIDEO_FOLDER, f"{video_id}{VIDEO_EXTENSION}")
                        frame_tensor = load_video_frames(video_path, NUM_FRAMES_TO_SAMPLE)

                    if frame_tensor is None:
                        frame_tensor = torch.zeros(NUM_FRAMES_TO_SAMPLE, 3, 224, 224).to(DEVICE)

                    visual_output = vision_model(frame_tensor)["image_embeds"]
                    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)
                    visual_output = torch.mean(visual_output, dim=0)
                    visual_output = visual_output / visual_output.norm(dim=-1, keepdim=True)

                    video_emb = visual_output.unsqueeze(0).detach().cpu().numpy()
                    video_embedding_cache[video_id] = video_emb

                score = 1 if video_id == gt_id else 0
                combined_features = np.hstack((text_emb, video_emb))
                dataset_pairs.append((combined_features, score))

    return dataset_pairs

def run_extraction(input_path, output_path, text_model, tokenizer, vision_model, label):
    if not os.path.exists(input_path):
        print(f"Skipping: {input_path} not found.")
        return

    with open(input_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Total {label} queries: {len(lines)}")
    dataset = process_lines(lines, text_model, tokenizer, vision_model, pbar_desc=label)

    with open(output_path, "wb") as f_out:
        pickle.dump(dataset, f_out)
    print(f"Successfully saved {label} dataset to {output_path}")

if __name__ == "__main__":
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_NAME)
    text_model = CLIPTextModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    vision_model = CLIPVisionModelWithProjection.from_pretrained(MODEL_NAME).to(DEVICE)
    text_model.eval()
    vision_model.eval()

    run_extraction(TRAIN_JSONL_PATH, TRAIN_OUTPUT_PATH, text_model, tokenizer, vision_model, "Train")

    run_extraction(VAL_JSONL_PATH, VAL_OUTPUT_PATH, text_model, tokenizer, vision_model, "Validation")

    run_extraction(TEST_JSONL_PATH, TEST_OUTPUT_PATH, text_model, tokenizer, vision_model, "Test")

    print("\nAll datasets have been processed and saved.")