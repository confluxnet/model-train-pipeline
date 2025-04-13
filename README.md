# Alibaba-NLP/gte-reranker-modernbert-base Fine-Tuning with BCE Loss

This repository provides a training pipeline to fine-tune the [Alibaba-NLP/gte-reranker-modernbert-base](https://huggingface.co/Alibaba-NLP/gte-reranker-modernbert-base) model on the [wryta/buidlaiv01](https://huggingface.co/wryta/buidlaiv01) dataset for reranking tasks using Binary Cross-Entropy Loss. The pipeline leverages the [sentence-transformers](https://www.sbert.net/) library with hard negative mining and multiple evaluation strategies.

## Overview

The training script performs the following steps:
- **Dataset Loading**: Downloads and splits the buidlaiv01 dataset.
- **Hard Negative Mining**: Uses a static retrieval model to mine hard negatives.
- **Training**: Fine-tunes the Alibaba-NLP/gte-reranker-modernbert-base model using Binary Cross-Entropy Loss.
- **Evaluation**: Evaluates the model with both CrossEncoderNanoBEIREvaluator and CrossEncoderRerankingEvaluator.
- **Model Saving and Upload**: Saves the final model locally and optionally uploads it to the Hugging Face Hub.

## Repository Structure

```
─ train.py           # Main training script
```

## Requirements

- **Python:** 3.7+
- **PyTorch:** As required by your hardware (CPU or GPU)
- **Datasets:** [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- **Sentence-Transformers:** For model loading and training
- **FAISS:** For efficient hard negative mining

## Installation

Clone the repository and navigate into it:

```bash
git clone https://github.com/confluxnet/model-train-pipeline.git
cd your-repo-name
```

## Usage

Run the training script with:

```bash
python train.py
```

The script will:
- Load and split the buidlaiv01 dataset.
- Apply hard negative mining.
- Fine-tune the model using Binary Cross-Entropy Loss.
- Evaluate using the provided evaluators.
- Save the final model locally and attempt to push it to the Hugging Face Hub (ensure you are logged in using `huggingface-cli login`).

## Training Configuration

Key parameters that can be adjusted directly in the script:
- **Model Name:** `"Alibaba-NLP/gte-reranker-modernbert-base"`
- **Batch Size:** `64` (for both training and evaluation)
- **Epochs:** `128`
- **Number of Hard Negatives:** `6`
- **Learning Rate and Warmup Ratio:** Configurable in the training arguments

These parameters are defined in the script and can be modified to fit your specific training requirements.

## Evaluation

The training pipeline uses two evaluators:
- **CrossEncoderNanoBEIREvaluator:** A lightweight evaluator using datasets such as MSMARCO, NFCorpus, and NQ.
- **CrossEncoderRerankingEvaluator:** Evaluates performance on reranking by mining hard negatives from the evaluation dataset.

Evaluator outputs are displayed during and after training to give insights into model performance.

3. Ensure your repository settings allow pushing new models.

## Contributing

Contributions and improvements are welcome! Please open issues or submit pull requests for any bugs or feature enhancements.

## License

This project is licensed under the Apache-2.0 License.
