# Qwen2.5-Omni Video Classification Pipeline

## Overview

This project provides a complete training pipeline to fine-tune Alibaba's Qwen2.5-Omni-7B multimodal model for a video classification task. We generate a synthetic video dataset (in the style of Kinetics, with one action label per video) and use Low-Rank Adaptation (LoRA) for efficient fine-tuning. The code integrates tightly with Hugging Face libraries for model and dataset handling, uses Hydra/OmegaConf for configuration management, and logs training progress to Weights & Biases (wandb). A simple CLI demo script is included to classify new videos using the trained model.

## Project Structure

- `README.md` – Overview, setup instructions, and usage examples (you're reading it).
- `pyproject.toml` – Required dependencies (Python libraries) to run the project.
- `config.yaml` – Configuration file (managed by Hydra) specifying dataset parameters, model hyperparameters, training settings, wandb logging, etc.
- `generator.py` – Synthetic video dataset generation and augmentation (PyTorch Dataset).
- `train.py` – Training script to fine-tune Qwen2.5-Omni-7B on the synthetic dataset using LoRA, with live logging to wandb.
- `demo.py` – Inference script to load the fine-tuned model and predict the label for a given video file.

## Setup Instructions

### Environment
Ensure you have Python 3.8+ and PyTorch installed (with CUDA if using GPU). An NVIDIA GPU with ~16 GB VRAM (e.g., RTX 3080) is the minimal requirement for training. For larger GPUs (e.g., 80 GB A100), you can adjust the configuration to utilize more resources (e.g., disable 4-bit quantization, use higher resolution frames, larger batch size).

### Install Dependencies
Install necessary packages in your pip environment:
```bash
pip install -e .
```
Note: The Qwen2.5-Omni model is a cutting-edge model not yet in the stable transformers release. The code assumes a version of HF Transformers that supports Qwen2.5-Omni. If you encounter a KeyError or missing model error, install transformers from source as indicated in Qwen's documentation:
```bash
pip install git+https://github.com/huggingface/transformers@f742a644ca32e65758c3adb36225aef1731bd2a8
pip install accelerate
```


### Weights & Biases Setup (optional)
The training script logs metrics to wandb. If you want to use this:
```bash
wandb login
```
(or set the WANDB_API_KEY environment variable).

By default, wandb logging is in online mode. To disable or run offline, you can adjust `wandb.mode` in the config or set the environment variable:
```bash
WANDB_MODE=disabled
```

## Usage

### Training the Model
After installing dependencies, run the training script. The configuration is managed by Hydra, so you can override any config option via the command line if desired.

Basic training command:
```bash
python train.py
```
This will use the default settings in `config.yaml` to generate a synthetic dataset and fine-tune the model. Training progress (loss per batch, loss/accuracy per epoch) will be printed and logged to wandb (if enabled).

Examples of overriding config options:
- Train for more epochs and use a different wandb run name:
  ```bash
  python train.py train.num_epochs=5 wandb.run_name="qwen_lora_exp5"
  ```
- Increase the number of synthetic videos or classes (be mindful of memory):
  ```bash
  python train.py dataset.num_videos=500 dataset.num_classes=8
  ```
- If using a larger GPU (e.g., A100 80GB) and you want higher precision and larger input size:
  ```bash
  python train.py model.use_4bit=false train.precision=bf16 dataset.frame_height=224 dataset.frame_width=224
  ```
  (This disables 4-bit quantization, uses bfloat16 precision, and sets video frame resolution to 224x224.)

The model and LoRA weights are saved at the end of training (default output directory is specified in the config, e.g., `qwen_video_lora/`). After training, you should see a directory (e.g., `qwen_video_lora`) containing the adapter weights and configuration.

### Demo: Video Classification Inference
Use `demo.py` to run the trained model on a new video and get a predicted label. For example:
```bash
python demo.py demo.video_path="path/to/your_video.mp4"
```
The script will load the base Qwen2.5-Omni-7B model and apply the fine-tuned LoRA weights from the training output (by default it looks for the adapter in the `qwen_video_lora/` directory). It will then print the predicted class label for the video.

Note: The synthetic dataset in this example consists of simple moving-dot patterns. The model fine-tunes to recognize these specific patterns. For a real-world use, you would replace the synthetic data generation with your actual video dataset (e.g., processed frames from videos and labels) and fine-tune similarly. This project is meant to showcase the pipeline structure (data, training, inference) rather than achieve state-of-the-art accuracy on real video data.
