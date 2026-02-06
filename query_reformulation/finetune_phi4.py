import torch
import torch.nn as nn
import pandas as pd
import os
import re
import numpy as np
from transformers import (
    BertTokenizer, 
    BertModel, 
    TrainerCallback,
    AutoModelForCausalLM, 
    AutoTokenizer
)
from peft import LoraConfig, get_peft_model
from trl import OnlineDPOTrainer, OnlineDPOConfig
from datasets import load_dataset
from tqdm import tqdm

TRAIN_FILE = "/root/VQPP/rephrasing/train_dpo.csv" 
VAL_FILE   = "/root/VQPP/rephrasing/validation_dpo.csv"      
TEST_FILE  = "/root/VQPP/resources/GRAM/metrics/msrvtt_test.csv"                
REWARD_MODEL_PATH = "/root/VQPP/baselines/finetune_bert/best_model.pth"
OUTPUT_DIR = "/root/VQPP/rephrasing/phi-4-reformulator-final"

MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1e-6
BATCH_SIZE = 4
GRAD_ACCUMULATION = 8
BETA = 0.3                    
SAVE_STEPS = 50

def clean_generated_text(text):
    text = text.strip()

    if '" or "' in text or "' or '" in text:
        text = text.split(' or ')[0]

    if '": "' in text or "': '" in text:
        matches = re.findall(r'[:=]\s*["\']([^"\']+)["\']', text)
        if matches:
            return " ".join(matches)

    if text.startswith('{') and text.endswith('}'):
        text = text[1:-1]

    lines = [line.strip() for line in text.split('\n') if line.strip()]
    
    for line in reversed(lines):
        if re.match(r'^(Output|Reformulated|Query|Answer|Result)[:\s]*$', line, flags=re.IGNORECASE):
            continue
            
        line = re.sub(
            r'^(?:Input|Query|Reformulated|Search(?: for)?|Find(?: images of)?|Image of|Context|Subject|Entities|Keywords)\s*[:|-]?\s*', 
            '', 
            line, 
            flags=re.IGNORECASE
        )
        
        line = line.strip().strip('"').strip("'")
        
        if line:
            return line

    return text.strip()

class BertRegressor(nn.Module):
    def __init__(self, n_outputs=1):
        super(BertRegressor, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-cased")
        self.linear1 = nn.Linear(self.bert.config.hidden_size, 512)
        self.linear2 = nn.Linear(512, n_outputs)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, input_ids, attention_mask):
        _, pooled_output = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=False)
        x = self.linear1(pooled_output)
        x = self.relu(x)
        x = self.drop(x)
        x = self.linear2(x)
        return x

class CustomBertJudge:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def get_scores(self, texts):
        inputs = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=512
        ).to(self.device)
        with torch.no_grad():
            scores = self.model(inputs['input_ids'], inputs['attention_mask'])
        return scores.flatten().cpu().tolist()

    def judge(self, prompts, completions, **kwargs):
        flat_completions = [c for pair in completions for c in pair]
        
        cleaned_queries = [clean_generated_text(str(c)) for c in flat_completions]
        
        flat_scores = self.get_scores(cleaned_queries)

        decision_probs = []
        for i in range(0, len(flat_scores), 2):
            if flat_scores[i] > flat_scores[i+1]:
                decision_probs.append(1.0)
            else:
                decision_probs.append(0.0)
        return decision_probs

class ValidationScoreCallback(TrainerCallback):
    def __init__(self, val_csv_path, tokenizer, judge):
        self.val_df = pd.read_csv(val_csv_path)
        self.tokenizer = tokenizer
        self.judge = judge
        self.prompts = []
        sys_msg = "Reformulate the user query to maximize retrieval. Output ONLY the reformulated query."
        for q in self.val_df['query']: 
            txt = f"<|system|>\n{sys_msg}<|end|>\n<|user|>\n{q}<|end|>\n<|assistant|>\n"
            self.prompts.append(txt)

    def on_step_end(self, args, state, control, model, **kwargs):
        if state.global_step % args.save_steps == 0 and state.global_step > 0:
            print(f"\n[Step {state.global_step}] Running Validation...")
            
            if hasattr(model, "module"):
                inference_model = model.module
            else:
                inference_model = model
            
            inference_model.eval()
            generated_queries = []
            
            with torch.no_grad():
                batch_size = 16
                for i in range(0, len(self.prompts), batch_size):
                    batch_prompts = self.prompts[i : i + batch_size]
                    inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                    outputs = inference_model.generate(
                        **inputs, max_new_tokens=128, use_cache=True, pad_token_id=self.tokenizer.pad_token_id
                    )
                    decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)

                    generated_queries.extend([clean_generated_text(d) for d in decoded])

            scores = self.judge.get_scores(generated_queries)
            avg_score = np.mean(scores)
            
            print(f"Validation Average Reward: {avg_score:.4f}")
            print(f"Example Gen: {generated_queries[0]}")
            print("-" * 30)
            inference_model.train()

class TestSetGenerationCallback(TrainerCallback):
    def __init__(self, test_csv_path, tokenizer, output_dir):
        self.test_df = pd.read_csv(test_csv_path)
        self.tokenizer = tokenizer
        self.output_dir = output_dir
        self.prompts = []
        sys_msg = "Reformulate the user query to maximize retrieval. Output ONLY the reformulated query."
        for q in self.test_df['Query']:
            txt = f"<|system|>\n{sys_msg}<|end|>\n<|user|>\n{q}<|end|>\n<|assistant|>\n"
            self.prompts.append(txt)

    def on_save(self, args, state, control, model, **kwargs):
        step = state.global_step
        print(f"\n[Step {step}] Saving Test Set CSV...")
        
        if hasattr(model, "module"):
            inference_model = model.module
        else:
            inference_model = model
            
        inference_model.eval()
        rephrased_queries = []
        batch_size = 32 
        
        with torch.no_grad():
            for i in tqdm(range(0, len(self.prompts), batch_size), desc="Generating Test Set"):
                batch_prompts = self.prompts[i : i + batch_size]
                inputs = self.tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to("cuda")
                outputs = inference_model.generate(
                    **inputs, max_new_tokens=128, use_cache=True, pad_token_id=self.tokenizer.pad_token_id
                )
                decoded = self.tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
                rephrased_queries.extend([clean_generated_text(d) for d in decoded])

        save_path = os.path.join(self.output_dir, f"test_rephrased_step_{step}.csv")
        result_df = self.test_df.copy()
        result_df['rephrased_query'] = rephrased_queries
        result_df.to_csv(save_path, index=False)
        print(f"Saved to: {save_path}")
        inference_model.train()

def train():
    model_id = "microsoft/Phi-4-mini-instruct"
    policy_tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    policy_tokenizer.padding_side = "left"
    policy_tokenizer.pad_token = policy_tokenizer.eos_token 

    policy_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="sdpa"
    )
    
    peft_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear"
    )
    policy_model = get_peft_model(policy_model, peft_config)
    policy_model.print_trainable_parameters()

    reward_tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    reward_model = BertRegressor()
    try:
        reward_model.load_state_dict(torch.load(REWARD_MODEL_PATH))
    except:
        reward_model = torch.load(REWARD_MODEL_PATH)
    reward_model.cuda().eval()

    bert_judge = CustomBertJudge(reward_model, reward_tokenizer)

    def format_prompt(example):
        sys_msg = "Reformulate the user query to maximize retrieval. Output ONLY the reformulated query."

        one_shot = (
            "<|user|>\nfunny cat videos<|end|>\n"
            "<|assistant|>\ncompilation of funny cats playing<|end|>\n"
        )
        return {
            "prompt": (
                f"<|system|>\n{sys_msg}<|end|>\n"
                f"{one_shot}"
                f"<|user|>\n{example['query']}<|end|>\n"
                f"<|assistant|>\n"
            )
        }

    ds_train = load_dataset("csv", data_files=TRAIN_FILE, split="train").map(format_prompt)

    val_callback = ValidationScoreCallback(VAL_FILE, policy_tokenizer, bert_judge)
    test_callback = TestSetGenerationCallback(TEST_FILE, policy_tokenizer, OUTPUT_DIR)

    args = OnlineDPOConfig(
        output_dir=OUTPUT_DIR,
        run_name="phi4-reformulator-final",
        
        per_device_train_batch_size=BATCH_SIZE,        
        gradient_accumulation_steps=GRAD_ACCUMULATION,
        gradient_checkpointing=True,
        
        bf16=True,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=SAVE_STEPS,
        save_total_limit=10,
        report_to="none",
        eval_strategy="no",
        
        max_new_tokens=128,
        max_length=MAX_SEQ_LENGTH,
    )

    trainer = OnlineDPOTrainer(
        model=policy_model,
        judge=bert_judge,
        processing_class=policy_tokenizer,
        train_dataset=ds_train,
        args=args,
        callbacks=[val_callback, test_callback]
    )

    trainer.train()
    
    policy_model.save_pretrained(f"{OUTPUT_DIR}/final_model")
    policy_tokenizer.save_pretrained(f"{OUTPUT_DIR}/final_model")

if __name__ == "__main__":
    train()