import os
from tqdm import tqdm
import numpy as np
from jiwer import wer, Compose, ToLowerCase, RemoveMultipleSpaces, Strip
from openai import OpenAI
import torch
import string
from nltk.stem import WordNetLemmatizer
from datasets import load_dataset, Dataset, Audio
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer,
    pipeline,
)
import torch.nn.functional as F
import key_gpt

# ===========================
# Text preprocessing tools (unchanged)
# ===========================
lemmatizer = WordNetLemmatizer()
STOPWORDS = {
    "that", "and", "-", "the", "an", "as", "or", "this", "with", "have", "so", "a", "my"
}

def lemmatize_text(text):
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return ' '.join(tokens)

def remove_stopwords(text):
    tokens = text.split()
    tokens = [w for w in tokens if w not in STOPWORDS]
    return ' '.join(tokens)

def punctuation_to_space(text):
    table = str.maketrans({p: " " for p in string.punctuation})
    return text.translate(table)

transform = Compose([
    ToLowerCase(),
    punctuation_to_space,
    RemoveMultipleSpaces(),
    Strip(),
    remove_stopwords,
    lemmatize_text,
])

# ===========================
# Enhanced Configuration
# ===========================
CONFIG = {
    "use_hf": True,
    "hf_dataset": "leduckhai/MultiMed",
    "hf_config": "English",
    "hf_split": "train",
    "max_files": 100,
    "output_dir": "whisper-dpo-finetuned-fixed",
    "api_key": key_gpt.api_key,  # Use the key from key_gpt.py
    "context": "General English conversation.",
    "batch_size": 4,
    "epochs": 20,
    "max_test_files": 200,
    "n_hypotheses": 4,
    "score_threshold": 0.1,
    "use_augmentation": True,
    "medical_weight": 0.7,
    "umls_terms_file": "mrconso_terms.txt",  # Path to the terms file
}
MAX_AUDIO_LENGTH = 20

# ===========================
# Load UMLS terms once at startup
# ===========================
UMLS_TERMS = set()
try:
    print(f"Loading UMLS terms from {CONFIG['umls_terms_file']}...")
    with open(CONFIG["umls_terms_file"], 'r', encoding='utf-8') as f:
        UMLS_TERMS = set(line.strip() for line in f if line.strip())
    print(f"Loaded {len(UMLS_TERMS)} terms from {CONFIG['umls_terms_file']}")
except FileNotFoundError:
    print(f"Error: {CONFIG['umls_terms_file']} not found. Term lookups will return False.")
except Exception as e:
    print(f"Error loading {CONFIG['umls_terms_file']}: {e}. Term lookups will return False.")

# ===========================
# Initialize models
# ===========================
client = OpenAI(api_key=CONFIG["api_key"], base_url="https://api.bianxie.ai/v1")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small").to(device)

asr_pipe = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-small",
    device=0 if device == "cuda" else -1,
    #chunk_length_s=30,
    return_timestamps=False,
    generate_kwargs={"language": "en"},
)

# ===========================
# Dataset loading functions (unchanged)
# ===========================
def load_local_dataset(audio_dir, transcript_dir, max_files=20):
    pairs = []
    files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav')])[:max_files]
    for fname in files:
        audio_path = os.path.join(audio_dir, fname)
        transcript_path = os.path.join(transcript_dir, os.path.splitext(fname)[0] + '.txt')
        if not os.path.exists(transcript_path):
            continue
        with open(transcript_path, 'r', encoding='utf-8') as f:
            transcript = f.read().strip()
        pairs.append({'audio': audio_path, 'transcript': transcript})
    return Dataset.from_list(pairs).cast_column("audio", Audio())

def load_hf_dataset(name, config, split, max_files=20):
    dataset = load_dataset(name, config, split=split)
    keys = list(dataset.features.keys())
    if 'transcription' in keys:
        dataset = dataset.rename_column('transcription', 'transcript')
    elif 'sentence' in keys:
        dataset = dataset.rename_column('sentence', 'transcript')
    elif 'text' in keys:
        dataset = dataset.rename_column('text', 'transcript')
    elif 'transcript' not in keys:
        raise ValueError(f"No transcript column found in dataset features: {keys}")
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))
    dataset = dataset.filter(lambda x: x['transcript'] and x['audio'] is not None)
    dataset = dataset.select(range(min(max_files, len(dataset))))
    return dataset

# ===========================
# Enhanced medical term extraction with caching
# ===========================
medical_term_cache = {}

def extract_medical_terms(text, client):
    prompt = (
        "Extract all MEDICAL RELATED terms from the following transcript. "
        "Return a python list of terms (one word or phrase per item), do not explain. "
        "If no medical terms are found, return an empty list [], DONOT return None. "
        f"Transcript: \"{text}\""
    )
    try:
        resp = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0,
        )
        import ast
        content = resp.choices[0].message.content
        if content == None:
            content = ""
        content = content.strip()
        try:
            terms = ast.literal_eval(content)
            assert isinstance(terms, list)
        except Exception:
            terms = []
        return [t.strip() for t in terms if t]
    except Exception as e:
        print(f"extract_medical_terms exception: {e}")
        return []

def check_umls_terms(terms):
    """
    Check if terms exist in the preloaded UMLS terms set
    """
    # Use the global UMLS_TERMS set for lookups
    results = {term: term in UMLS_TERMS for term in terms}
    return results

# ===========================
# Enhanced scoring function
# ===========================
def gpt_score(context, hypothesis):
    prompt = (
        f"You are an expert ASR evaluator. Context: {context}\n"
        f"Transcript: \"{hypothesis}\"\n"
        f"Rate the transcription quality (accuracy and context) from 0 (poor) to 1 (excellent). "
        "Just return the score as a number."
    )
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1,
        temperature=0,
    )
    score_str = response.choices[0].message.content.strip()
    try:
        score = float(score_str)
    except Exception:
        score = 1
    return score
def enhanced_score(reference, hypothesis, client):
    """
    Combines medical F1 score with general WER and confidence
    """
    # Medical F1 score
    pred_terms = extract_medical_terms(hypothesis, client)
    ref_terms = extract_medical_terms(reference, client)
    pred_hits = [t for t, ok in check_umls_terms(pred_terms).items() if ok]
    ref_hits = [t for t, ok in check_umls_terms(ref_terms).items() if ok]
    
    pred_set = set([t.lower() for t in pred_hits])
    ref_set = set([t.lower() for t in ref_hits])
    num_correct = len(pred_set & ref_set)
    num_pred = len(pred_set)
    num_ref = len(ref_set)
    
    precision = num_correct / num_pred if num_pred > 0 else 1.0
    recall = num_correct / num_ref if num_ref > 0 else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # General WER
    norm_hyp = hypothesis
    norm_ref = reference
    general_wer = wer(norm_ref, norm_hyp)
    wer_score = 1.0 - general_wer  # Convert to score (higher is better)
    
    # Length penalty (penalize too short/long outputs)
    len_ratio = len(hypothesis.split()) / (len(reference.split()) + 1e-6)
    len_penalty = np.exp(-abs(np.log(len_ratio)))
    
    # Combined score
    medical_weight = CONFIG["medical_weight"]
    combined_score = (
        medical_weight * f1 + 
        (1 - medical_weight) * wer_score * len_penalty
    )
    # GPT-based score
    gpt_score_value = gpt_score(CONFIG["context"], hypothesis)
    combined_score = (combined_score + gpt_score_value) / 2.0

    return combined_score

# ===========================
# Enhanced hypothesis generation
# ===========================
def generate_diverse_hypotheses(audio_array, n_hyps=12):
    """
    Generate hypotheses using various decoding strategies
    """
    hyps = []
    
    # Temperature sampling
    temperatures = [0.01, 0.2, 0.4, 0.6, 0.8, 1.0]
    for temp in temperatures[:n_hyps//2]:
        result = asr_pipe(
            audio_array,
            generate_kwargs={
                "temperature": temp,
                "do_sample": True,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3,
            }
        )
        hyps.append(result['text'])
    
    # Beam search with different beam sizes
    beam_sizes = [2, 4, 6, 8]
    for beam in beam_sizes[:n_hyps//3]:
        result = asr_pipe(
            audio_array,
            generate_kwargs={
                "num_beams": beam,
                "do_sample": False,
                "repetition_penalty": 1.1,
            }
        )
        hyps.append(result['text'])
    
    # Top-p (nucleus) sampling
    top_p_values = [0.9, 0.95, 0.99]
    for p in top_p_values[:n_hyps - len(hyps)]:
        result = asr_pipe(
            audio_array,
            generate_kwargs={
                "do_sample": True,
                "top_p": p,
                "temperature": 0.7,
                "repetition_penalty": 1.15,
            }
        )
        hyps.append(result['text'])
    import random
    # return hyps[:n_hyps]
    output_hyps = random.sample(hyps, min(n_hyps, len(hyps))) 
    print(f"Generated {len(output_hyps)} diverse hypotheses")
    return output_hyps # Ensure we return exactly n_hyps
# ===========================
# Audio augmentation
# ===========================
def augment_audio(audio, sampling_rate=16000):
    """
    Apply simple audio augmentation
    """
    augmented = []
    
    # Original
    augmented.append(audio)
    
    if CONFIG["use_augmentation"]:
        # Add white noise
        noise_level = 0.005
        noise = np.random.normal(0, noise_level, audio.shape)
        augmented.append(audio + noise)
        
        # Speed perturbation (simple resampling)
        speed_factor = 1.1
        indices = np.arange(0, len(audio), speed_factor)
        indices = indices[indices < len(audio)].astype(int)
        augmented.append(audio[indices])
    
    return augmented

# ===========================
# Enhanced DPO pair creation
# ===========================
def create_enhanced_dpo_pairs(dataset, client):
    dpo_pairs = []
    for idx, sample in enumerate(tqdm(dataset, total=len(dataset), desc="Processing DPO pairs")):
        audio = sample['audio']['array']
        if len(audio) > MAX_AUDIO_LENGTH * 16000:
            continue
        
        target = sample['transcript']
        
        augmented_audios = augment_audio(audio)
        
        for aug_idx, aug_audio in enumerate(augmented_audios):
            hyps = generate_diverse_hypotheses(aug_audio, n_hyps=CONFIG["n_hypotheses"])
            scores = [enhanced_score(target, h, client) for h in hyps]
            
            sorted_indices = np.argsort(scores)[::-1]
            for i in range(min(3, len(hyps)//2)):
                best_idx = sorted_indices[i]
                worst_idx = sorted_indices[-(i+1)]
                if scores[best_idx] - scores[worst_idx] >= CONFIG["score_threshold"]:
                    dpo_pairs.append({
                        'audio': aug_audio,
                        'ref': target,
                        'best': target,
                        'worst': hyps[worst_idx],
                        'score_diff': scores[best_idx] - scores[worst_idx]
                    })
    dpo_pairs.sort(key=lambda x: x['score_diff'], reverse=True)
    print(f"Created {len(dpo_pairs)} DPO pairs")
    return Dataset.from_list(dpo_pairs)

# ===========================
# Load datasets
# ===========================
if CONFIG["use_hf"]:
    print(f"Loading Hugging Face train split: {CONFIG['hf_dataset']} / {CONFIG['hf_config']} / {CONFIG['hf_split']}")
    dataset = load_hf_dataset(
        CONFIG["hf_dataset"],
        CONFIG["hf_config"],
        CONFIG["hf_split"],
        CONFIG["max_files"]
    )
    print(f"Loading Hugging Face test split: {CONFIG['hf_dataset']} / {CONFIG['hf_config']} / test")
    raw_test_dataset = load_hf_dataset(
        CONFIG["hf_dataset"],
        CONFIG["hf_config"],
        "test",
        CONFIG["max_test_files"]
    )
else:
    print(f"Loading local dataset from {CONFIG['audio_dir']} and {CONFIG['transcript_dir']}")
    dataset = load_local_dataset(CONFIG["audio_dir"], CONFIG["transcript_dir"], CONFIG["max_files"])
    raw_test_dataset = None

# ===========================
# Create enhanced DPO training set
# ===========================
print("Creating enhanced DPO pairs...")
dpo_dataset = create_enhanced_dpo_pairs(dataset, client)

# ===========================
# Create test dataset (unchanged structure)
# ===========================
test_dpo_pairs = []
print("Converting test split to dpo-pair format...")
for sample in tqdm(raw_test_dataset, total=len(raw_test_dataset), desc="Processing test DPO pairs"):
    audio = sample['audio']['array']
    if len(audio) > MAX_AUDIO_LENGTH * 16000:
        print(f"Skip too long audio: {len(audio)/16000:.1f}s")
        continue
    target = sample['transcript']
    test_dpo_pairs.append({
        'audio': audio,
        'ref': target,
        'best': target,
        'worst': target,
    })
test_dpo_dataset = Dataset.from_list(test_dpo_pairs)

# ===========================
# WER evaluation function (unchanged)
# ===========================
def compute_wer(dataset, model, processor):
    pred_texts = []
    ref_texts = []

    def safe_audio(x):
        if isinstance(x, np.ndarray):
            return x.astype(np.float32)
        elif isinstance(x, torch.Tensor):
            return x.cpu().numpy().astype(np.float32)
        elif isinstance(x, list):
            return np.array(x, dtype=np.float32)
        else:
            raise ValueError(f"Unknown audio type: {type(x)}")

    for sample in tqdm(dataset, desc="Computing WER"):
        audio = safe_audio(sample['audio'])
        if len(audio) > MAX_AUDIO_LENGTH * 16000:
            print(f"Skip too long audio: {len(audio)/16000:.1f}s")
            continue
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt",return_attention_mask=True)
        #input_features = inputs.input_features.to(model.device)
        predicted_ids = model.generate(
            input_features=inputs["input_features"].to(model.device),
            attention_mask=inputs["attention_mask"],
            language="<|en|>",
            task="transcribe",
            no_repeat_ngram_size=2,
            max_new_tokens=256,
        )
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        pred_texts.append(transcription)
        ref_texts.append(sample['ref'])
        print("REF:", sample['ref'])
        print("PRED:", transcription)
        print("NORM_REF:", transform(sample['ref']))
        print("NORM_PRED:", transform(transcription))

    norm_pred_texts = [transform(p) for p in pred_texts]
    norm_ref_texts = [transform(r) for r in ref_texts]
    return wer(norm_ref_texts, norm_pred_texts)

print("Calculating WER before finetuning...")
wer_before = compute_wer(test_dpo_dataset, model, processor)
print(f"WER before finetune: {wer_before:.4f}")

# ===========================
# Enhanced preprocessing
# ===========================
def preprocess(batch):
    audio = batch["audio"]
    input_features = processor(audio, sampling_rate=16000, return_tensors="pt",return_attention_mask=True).input_features.squeeze(0)
    labels = processor.tokenizer(batch["best"]).input_ids
    return {"input_features": input_features.tolist(), "labels": labels}

print("Preprocessing data for finetuning...")
finetune_dataset = dpo_dataset.map(preprocess)

# ===========================
# Data Collator
# ===========================
class WhisperContrastiveCollator:
    def __call__(self, features):
        input_features = []
        for f in features:
            x = f['input_features']
            x = torch.tensor(x)
            if len(x.shape) == 3 and x.shape[0] == 1:
                x = x.squeeze(0)
            input_features.append(x)
        input_features = torch.stack(input_features)
        labels = [torch.tensor(f['labels']) for f in features]
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
        return {'input_features': input_features, 'labels': labels}

# ===========================
# Enhanced Training Arguments
# ===========================
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["batch_size"],
    num_train_epochs=CONFIG["epochs"],
    fp16=torch.cuda.is_available(),
    save_strategy="epoch",
    logging_steps=10,
    report_to=[],
    gradient_accumulation_steps=4,
)

# ===========================
# Enhanced Trainer with contrastive loss
# ===========================
class WhisperContrastiveTrainer(Trainer):
    def __init__(self, *args, margin=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.margin = margin
        
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_features = inputs["input_features"].to(device)
        labels = inputs["labels"].to(device)
        negative_labels = inputs.get("negative_labels", None)
        
        # Positive loss
        pos_outputs = model(input_features=input_features, labels=labels)
        pos_loss = pos_outputs.loss
        
        # Contrastive loss if negative labels provided
        if negative_labels is not None:
            # Detach input features to create a new computation graph
            input_features_detached = input_features.detach().requires_grad_(True)
            negative_labels = negative_labels.to(device)
            neg_outputs = model(input_features=input_features_detached, labels=negative_labels)
            neg_loss = neg_outputs.loss
            
            # Margin loss: we want pos_loss < neg_loss - margin
            contrastive_loss = F.relu(self.margin - (neg_loss - pos_loss))
            total_loss = pos_loss + 0.1 * contrastive_loss
        else:
            total_loss = pos_loss
            
        return (total_loss, pos_outputs) if return_outputs else total_loss
trainer = WhisperContrastiveTrainer(
    model=model,
    args=training_args,
    train_dataset=finetune_dataset,
    data_collator=WhisperContrastiveCollator(),
    margin=0.5,
)

# ===========================
# Train and save model
# ===========================
print("Starting enhanced finetuning...")
trainer.train()

print(f"Saving finetuned model to {CONFIG['output_dir']} ...")
trainer.save_model(CONFIG["output_dir"])
processor.save_pretrained(CONFIG["output_dir"])

# ===========================
# Evaluate after finetuning
# ===========================
print("Calculating WER after finetuning...")
finetuned_model = WhisperForConditionalGeneration.from_pretrained(CONFIG["output_dir"]).to(device)
finetuned_model.config.forced_decoder_ids = None
finetuned_model.config.suppress_tokens = []
finetuned_model.generation_config.forced_decoder_ids = None
finetuned_model.generation_config._from_model_config = True

wer_after = compute_wer(test_dpo_dataset, finetuned_model, processor)
print(f"WER after finetune: {wer_after:.4f}")

# ===========================
# Medical-specific WER
# ===========================
'''def compute_medical_wer(dataset, model, processor, client):
    """Compute WER specifically for medical terms"""
    medical_pred = []
    medical_ref = []
    
    for sample in tqdm(dataset[:50], desc="Computing Medical WER"):
        audio = sample['audio']
        if isinstance(audio, dict):
            audio = audio['array']
        
        # Get prediction
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        input_features = inputs.input_features.to(model.device)
        predicted_ids = model.generate(input_features)
        pred_text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        
        # Extract medical terms only
        pred_medical = extract_medical_terms(pred_text, client)
        ref_medical = extract_medical_terms(sample['ref'], client)
        
        medical_pred.append(" ".join(pred_medical))
        medical_ref.append(" ".join(ref_medical))
    
    return wer(medical_ref, medical_pred)'''

#print("\nComputing medical-specific WER...")
#medical_wer_before = compute_medical_wer(test_dpo_dataset[:50], model, processor, client)
#medical_wer_after = compute_medical_wer(test_dpo_dataset[:50], finetuned_model, processor, client)

print("\n--- Summary ---")
print(f"Overall WER before: {wer_before:.4f}")
print(f"Overall WER after:  {wer_after:.4f}")
#print(f"Medical WER before: {medical_wer_before:.4f}")
#print(f"Medical WER after:  {medical_wer_after:.4f}")
print(f"Finetuned model saved to: {CONFIG['output_dir']}")
