# Whisper DPO Finetuning for Medical ASR

## Overview
This project implements a finetuning pipeline for the Whisper-small model using Direct Preference Optimization (DPO) to improve Automatic Speech Recognition (ASR) performance, with a focus on medical terminology. It leverages a combination of medical term extraction, enhanced scoring, and audio augmentation to create high-quality DPO pairs for training.

## Features
- **Dataset**: Supports Hugging Face datasets (`leduckhai/MultiMed`) or local audio/transcript pairs.
- **Medical Term Extraction**: Uses UMLS terms and GPT-4o to identify medical terms for scoring.
- **Enhanced Scoring**: Combines medical F1 score, WER, length penalty, and GPT-based evaluation.
- **Audio Augmentation**: Applies noise and speed perturbation for robust training.
- **Diverse Hypotheses**: Generates varied transcriptions using temperature sampling, beam search, and top-p sampling.
- **Contrastive Training**: Implements a custom Trainer with contrastive loss for DPO.

## Requirements
- Python 3.8+
- Libraries: `torch`, `transformers`, `datasets`, `jiwer`, `nltk`, `numpy`, `tqdm`, `openai`
- UMLS terms file (`mrconso_terms.txt`) for medical term validation
- API key for GPT-4o (stored in `key_gpt.py`)

## Usage
1. **Setup**:
   - Install dependencies: `pip install -r requirements.txt`
   - Ensure `mrconso_terms.txt` is available or update `CONFIG["umls_terms_file"]`.
   - Set up `key_gpt.py` with your API key.

2. **Configuration**:
   - Modify `CONFIG` in `mixed_scoring.py` for dataset, training, and augmentation settings.

3. **Run**:
   ```bash
   python mixed_scoring.py
   ```
   This will load the dataset, create DPO pairs, finetune the model, and evaluate WER before and after training.

## Output
- Finetuned model saved to `whisper-dpo-finetuned-fixed/`.
- WER metrics for general and medical-specific performance (medical WER currently commented out).

## Notes
- Ensure audio files are ≤20 seconds to avoid skipping.
- Adjust `CONFIG["max_files"]` and `CONFIG["max_test_files"]` for dataset size.
- Medical WER evaluation is commented out but can be enabled for focused analysis.
