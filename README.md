# VQPP: Video Query Performance Prediction Benchmark

This repository contains the official code for the paper **"VQPP: Video Query Performance Prediction Benchmark"**.

VQPP is a benchmark designed to evaluate methods that estimate the difficulty of a text query for video retrieval systems. It standardizes the evaluation of **Pre-retrieval** and **Post-retrieval** QPP estimators across two datasets (MSR-VTT, VATEX) and two state-of-the-art retrieval architectures (GRAM, VAST).

The official data can be found [here](https://huggingface.co/datasets/funzon3/VQPP/tree/main). Please download the contents of the dataset and place them in the resources directory inside the repo.

##  Repository Structure

```text
.
├── baselines/                  # Source code for QPP baseline methods
│   ├── linguistics/            # Simple linguistic heuristics (Word count, POS tags)
│   ├── finetune_bert/          # Pre-retrieval estimator (BERT regressor)
│   ├── finetune_clip/          # Post-retrieval estimators (CLIP score, CLIP4CLIP)
│   ├── corelation_cnn/         # Score distribution analysis (Correlation CNN)
│   └── llm/                    # Zero-shot/Few-shot estimation via Llama 3.1
├── prep_datasets/              # Source code for downloading and processing the original video corpus
├── query_reformulation/        # Application: QPP-guided Query Reformulation (Phi-4 + DPO)
└── pyproject.toml              # Project dependencies and configuration
```
##  Installation

This project is managed using [Poetry](https://python-poetry.org/) and requires **Python 3.12+**.

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/VQPP.git](https://github.com/yourusername/VQPP.git)
   cd VQPP
    ```
2. Install dependencies with Poetry: This will automatically create a virtual environment and install all required packages (including PyTorch with CUDA support as specified in pyproject.toml).
```bash
   poetry install
```

## Downloading and processing the original video corpus

## MSR-VTT
MSR-VTT can be downloaded using this [link](https://cove.thecvf.com/datasets/839).

After downloading, unzip the files and make sure the paths inside prep_datasets/MSRVTT/process_msrvtt.py match the video files, then simply run the following command.

```bash
   poetry run python prep_datasets/MSRVTT/process_msrvtt.py
```

## VATEX

For VATEX we need to scrape the video from youtube. For this we provide the prep_datasets/VATEX/download_vatex.py script. This script will download all the available videos (Note: Some video might not be available on youtube anymore) and process them.

```bash
   poetry run python prep_datasets/VATEX/download_vatex.py
```

(Note: From our experience this process took several days.)

## Running Baselines
We provide implementations for three categories of QPP estimators.(Note: Please make sure that the paths in each script point to the correct files!)

### 1. Linguistic Baselines
Extracts statistical features (word count, POS tags, etc.) and correlates them with performance.
```bash
   poetry run python baselines/linguistics/linguistics.py
```
### 2. Pre-Retrieval Estimators
Finetuned BERT: Trains a BERT regressor to predict Reciprocal Rank or Recall@10 directly from the query text.
```bash
   poetry run python baselines/finetune_bert/finetune_bert.py
```
LLM (Llama 3.1): Uses In-Context Learning (Few-shot) to predict difficulty. (Note: Make sure you have created and added the apy_key.py file containing your Google AI Studio and Groq api keys!)
```bash
   poetry run python baselines/llm/llm.py
```
### 3. Post-Retrieval Estimators
Finetuned CLIP: Uses visual-semantic similarity of retrieved candidates.
```bash
   poetry run python baselines/finetune_clip/create_dataset_clip.py
   poetry run python baselines/finetune_clip/finetune_clip.py
   poetry run python baselines/finetune_clip/compute_corelation.py
```
Correlation CNN: Analyzes the correlation between the queries and the video embeddings.
```bash
   poetry run python baselines/corelation_cnn/create_dataset.py
   poetry run python baselines/corelation_cnn/train_cnn.py
   poetry run python baselines/corelation_cnn/compute_corelation.py
```
## Application: Query Reformulation
We demonstrate an active use-case of QPP by training an LLM to reformulate queries to maximize their predicted retrieval performance.

### Pipeline
- Reward Model: We use the Finetuned BERT (from the baselines) as the reward function.
- Policy Model: We fine-tune Phi-4-mini using Online DPO.

### How to Run

First, ensure you have a trained BERT checkpoint from the finetune_bert step. Then run the DPO training:
```bash
   poetry run python query_reformulation/finetune_phi4.py
```
## Citation

If you wish to use these resources, please cite the work as follows:

```text
@article{lutu2026vqppvideoqueryperformance,
      title={VQPP: Video Query Performance Prediction Benchmark}, 
      author={Adrian Catalin Lutu and Eduard Poesina and Radu Tudor Ionescu},
      year={2026},
      journal={arXiv preprint arXiv:2602.17814},
      url={https://arxiv.org/abs/2602.17814}, 
}
```
