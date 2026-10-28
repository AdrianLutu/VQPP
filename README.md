# VQPP: Video Query Performance Prediction Benchmark

This repository contains the official code for the paper **"VQPP: Video Query Performance Prediction Benchmark"**.

VQPP is a benchmark designed to evaluate methods that estimate the difficulty of a text query for video retrieval systems. It standardizes the evaluation of **Pre-retrieval** and **Post-retrieval** QPP estimators across two datasets (MSR-VTT, VATEX) and two state-of-the-art retrieval architectures (GRAM, VAST).

The official data can be found [here](https://huggingface.co/datasets/funzon3/VQPP/tree/main). Please download the contents of the dataset and place them in the resources directory inside the repository.

##  Repository Structure

```text
.
├── baselines/                  # Source code for QPP baseline methods
│   ├── linguistics/            # Simple linguistic heuristics (Word count, POS tags)
│   ├── finetune_bert/          # Pre-retrieval estimator (BERT regressor)
│   ├── finetune_clip/          # Post-retrieval estimators (CLIP score, CLIP4CLIP)
│   ├── corelation_cnn/         # Score distribution analysis (Correlation CNN)
│   ├── meta_svr/               # Meta Regression Model
|   └── llm/                    # Zero-shot/Few-shot estimation via Llama 3.1
|   
├── prep_datasets/              # Source code for downloading and processing the original video corpus
├── query_reformulation/        # Application: QPP-guided Query Reformulation (Phi-4 + DPO)
└── pyproject.toml              # Project dependencies and configuration
```
##  Installation

This project is managed using [Poetry](https://python-poetry.org/) and requires **Python 3.12+**.

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AdrianLutu/VQPP.git
   cd VQPP
    ```
2. Install dependencies with Poetry: This will automatically create a virtual environment and install all required packages (including PyTorch with CUDA support as specified in pyproject.toml).
```bash
   poetry install
```

3. Download the official dataset [here](https://huggingface.co/datasets/funzon3/VQPP/tree/main) and place the GRAM and VAST directories into the resources directory of the repository.

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
Finetuned BERT: Trains a BERT regressor to predict Reciprocal Rank or Recall@10 directly from the query text. Set `RETRIEVAL_MODEL`, `DATASET` and `METRIC` at the top of the script. The best model of the grid search is saved to `baselines/finetune_bert/checkpoints/bert_<RETRIEVAL_MODEL>_<DATASET>_<rr|r10>.pth`.
```bash
   poetry run python baselines/finetune_bert/finetune_bert.py
```
LLM (Llama 3.1): Uses In-Context Learning (Few-shot) to predict difficulty. The 16 in-context examples are the nearest training queries in the space of pre-trained BERT embeddings. Set `RETRIEVAL_MODEL`, `DATASET` and `METRIC` (`Recall@10` or `Reciprocal_Rank`) at the top of the script. (Note: Make sure you have created the `baselines/llm/api_key.py` file defining `API_KEY_GROQ` with your Groq API key!)
```bash
   poetry run python baselines/llm/llm.py
```
### 3. Post-Retrieval Estimators
Finetuned CLIP: Uses visual-semantic similarity of retrieved candidates.
```bash
   poetry run python baselines/finetune_clip/create_dataset_clip.py
   poetry run python baselines/finetune_clip/finetune_clip.py
   poetry run python baselines/finetune_clip/compute_correlations.py
```
Finetuned CLIP4Clip: Same classifier, but the frames of each video are aggregated by the temporal transformer of CLIP4Clip, trained together with the classifier. Point the paths of `compute_correlations.py` to the CLIP4Clip test dataset and predictions.
```bash
   poetry run python baselines/finetune_clip/create_dataset_clip4clip.py
   poetry run python baselines/finetune_clip/finetune_clip4clip.py
   poetry run python baselines/finetune_clip/compute_correlations.py
```
Correlation CNN: Analyzes the correlation between the embeddings of the top 25 videos retrieved for each query. `TARGET_METRIC` has to be set to the same value in the three scripts: every file they write carries the metric in its name.
```bash
   poetry run python baselines/corelation_cnn/create_dataset.py
   poetry run python baselines/corelation_cnn/train_cnn.py
   poetry run python baselines/corelation_cnn/compute_correlation.py
```
META-SVR: Uses the prediction of the fine-tuned BERT model and the fine-tuned CLIP model to predict the queries performance:

Firstly you will need to run and save the predctions for both models and point the paths to the apropriate files.
```bash
   poetry run python baselines/meta_svr/get_metrics.py
   poetry run python baselines/meta_svr/meta_svr.py
```

## Application: Query Reformulation
We demonstrate an active use-case of QPP by training an LLM to reformulate queries to maximize their predicted retrieval performance.

### Pipeline
- Reward Model: We use the Finetuned BERT (from the baselines) as the reward function.
- Policy Model: We fine-tune Phi-4-mini using Online DPO.

### How to Run

First, train the reward model: run the finetune_bert step with `RETRIEVAL_MODEL = "GRAM"`, `DATASET = "msrvtt"` and `METRIC = "Recall@10"`, which writes `baselines/finetune_bert/checkpoints/bert_GRAM_msrvtt_r10.pth`. Then build the DPO training set, run the DPO training, and pick, for every test query, the original or the reformulated version according to the predictor:
```bash
   poetry run python query_reformulation/create_dataset.py
   poetry run python query_reformulation/finetune_phi4.py
   poetry run python query_reformulation/chose_query.py
```
All outputs are written to `query_reformulation/outputs/`.

Note: the results of Table 5 in the paper were obtained with a learning rate of `1e-6`. The default is now `5e-6` (see the notes below); set `LEARNING_RATE = 1e-6` in `finetune_phi4.py` to reproduce Table 5.

## Notes on the paper and the implementation

Differences between the paper (arXiv:2602.17814) and the code, and inconsistencies within the paper:

- **Fine-tuned BERT is only partially fine-tuned.** The paper describes a fine-tuned BERT with a regression head on the [CLS] token. In the code, the embeddings and the first 11 encoder layers are frozen; only the last encoder layer, the pooler and the regression head are trained. The head reads the pooler output (the [CLS] state passed through BERT's dense + tanh layer), not the raw [CLS] state.
- **"Last convolutional block" in the paper.** The paper attaches the regression head "to the [CLS] token returned by the last convolutional block". BERT has no convolutional blocks; this should read "the last transformer block" (encoder layer), and, as above, the code actually uses the pooler output.
- **Early stopping.** The paper states that all models are trained with early stopping. The fine-tuned BERT now stops when the validation MSE has not improved for 3 epochs and keeps the best epoch; earlier versions of the script always trained for 12 epochs and kept the last one. The CLIP classifier and the Correlation CNN train for 25 epochs and keep the epoch with the lowest validation loss.
- **Correlation CNN dense layers.** The paper describes "two dense layers of 512 neurons each" after global average pooling. The code has one dense layer of 512 neurons (`fc1`, 512 → 512) followed by the 512 → 1 regression output.
- **CLIP4Clip temporal transformer.** The paper aggregates the frames with "the temporal transformer in CLIP4Clip". Earlier versions of the code used the mean of the 12 frame embeddings instead (the *meanP* variant of CLIP4Clip): the HuggingFace port `Searchium-ai/clip4clip-webvid150k` ships only the CLIP text and vision encoders, without the weights of a temporal transformer. `finetune_clip4clip.py` now implements the *seqTransf* header of CLIP4Clip: frame position embeddings and 4 transformer layers, initialised from the CLIP text encoder as in CLIP4Clip, a residual connection to the frame embeddings, then mean pooling. It is trained together with the classifier. The CLIP4Clip numbers of the ablation in the paper were obtained with the mean pooling, so they will differ from the numbers this version produces.
- **LLM in-context examples.** The paper selects the 16 examples with a k-NN over pre-trained BERT embeddings, which is what the script does now. Earlier versions of the script used Google's `text-embedding-004` instead, and supported only Recall@10; Reciprocal Rank is now supported as well.
- **Split sizes.** The released data (`resources/{GRAM,VAST}/metrics`) does not match Table 1 of the paper:

  | Dataset | Table 1 (train / val / test) | Released CSVs (train / val / test) |
  |---|---|---|
  | MSR-VTT | 24,490 / 879 / 7,374 | 29,307 / 875 / 7,331 |
  | VATEX | 13,872 / 714 / 3,469 | 13,789 / 713 / 3,459 |

  The MSR-VTT row of Table 1 also sums to 32,743, while the text states 32,732 queries. The "56K queries" of the paper match the released data (55,474 queries), not Table 1 (50,798).
- **VATEX videos.** The paper uses 41,250 VATEX videos. The annotation files in `prep_datasets/VATEX` cover the training (25,991) and validation (3,000) videos, and `download_vatex.py` downloads the training ones; change `ANNOTATION_FILE` and `BASE_DIR` to download the validation videos.
- **DPO learning rate.** Table 5 was obtained with a learning rate of `1e-6`, which is low for LoRA: the adapters start at zero and, with `lora_alpha` equal to `r`, the update is not scaled up, so one epoch barely moves the policy. The default is now `5e-6`, the rate of the DPO + LoRA recipes of the HuggingFace alignment handbook.
- **Phi-4-mini prompt format.** The prompts now follow the chat template of Phi-4-mini-instruct (`<|role|>content<|end|>`, with no newlines around the tags) and start with the BOS token that `OnlineDPOTrainer` adds to every training prompt, so training and generation see the same tokens. Earlier versions used the newline-separated Phi-3 format, and generated without the BOS token.
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
