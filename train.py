import os
import torch
from torch.utils.data import DataLoader, random_split
# from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor
from peft import LoraConfig, get_peft_model
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig

# Import the SyntheticVideoDataset from dataset.py
from dataset import SyntheticVideoDataset

# We define a custom collate function for DataLoader to handle a batch of video frames (lists of images)
def collate_fn(batch):
    """
    Collate function to combine a list of (frames, label) pairs into batch format.
    Returns:
      videos_batch: list of list of PIL images for each sample
      labels_batch: tensor of labels
    """
    videos_batch = [item[0] for item in batch]   # list of frame lists
    labels_batch = torch.tensor([item[1] for item in batch], dtype=torch.long)
    return videos_batch, labels_batch

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Pretty-print the config for verification
    print("Configuration:\n", OmegaConf.to_yaml(cfg))

    # Ensure we stay in the original working directory (because Hydra by default might change the cwd)
    os.chdir(hydra.utils.get_original_cwd() or ".")

    # Set up Weights & Biases logging
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, mode=cfg.wandb.mode,
               config=OmegaConf.to_container(cfg, resolve=True))
    
    # Generate synthetic dataset
    full_dataset = SyntheticVideoDataset(num_videos=cfg.dataset.num_videos,
                                        num_classes=cfg.dataset.num_classes,
                                        frames_per_video=cfg.dataset.frames_per_video,
                                        frame_height=cfg.dataset.frame_height,
                                        frame_width=cfg.dataset.frame_width,
                                        max_skip=cfg.dataset.max_skip,
                                        seed=cfg.dataset.seed)
    # Optionally, split into training and validation sets (80/20 split here)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(cfg.dataset.seed or 42))
    else:
        train_dataset = full_dataset
        val_dataset = None

    # Create DataLoader for training (and validation if applicable)
    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collate_fn)

    # Load the Qwen2.5-Omni model and processor
    model_name = cfg.model.base_model
    print(f"Loading base model '{model_name}'...")
    if cfg.model.use_8bit:
        # Load model in 8-bit precision (requires bitsandbytes)
        model = Qwen2_5OmniModel.from_pretrained(model_name, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16)
    else:
        # Load model in specified precision (fp16/bf16/fp32)
        dtype = torch.float32
        if cfg.train.precision == "fp16":
            dtype = torch.float16
        elif cfg.train.precision == "bf16":
            dtype = torch.bfloat16
        model = Qwen2_5OmniModel.from_pretrained(model_name, device_map="auto", torch_dtype=dtype)

    model = model.thinker  # the actual text+vision submodule
    model.train()     
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    # Prepare LoRA configuration and wrap the model with LoRA
    lora_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=list(cfg.model.lora_target_modules),  # list of module names to apply LoRA
        bias="none",
        task_type="CAUSAL_LM"  # we're fine-tuning a causal LM (decoder) model
    )
    model = get_peft_model(model, lora_config)
    model.train()  # set model to training mode

    # Print number of trainable parameters vs total parameters for confirmation
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Model loaded. Total parameters: {total_params}, Trainable (LoRA) parameters: {trainable_params}")

    # Set up optimizer (only parameters with requires_grad=True, which will be the LoRA parameters)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.learning_rate)

    # Training loop
    global_step = 0
    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_idx, (videos, labels) in enumerate(train_loader):
            # Prepare input for the model
            # All videos in the batch and corresponding system prompt text
            batch_size = len(videos)
            text_prompts = [cfg.train.system_prompt] * batch_size

            # Use the processor to tokenize text and preprocess video frames
            inputs = processor(text=text_prompts, videos=videos, return_tensors="pt", padding=True, truncation=True)
            # Move inputs to the model device and cast to model dtype
            inputs = inputs.to(model.device)
            # Extract input_ids and attention_mask from processed inputs
            input_ids = inputs["input_ids"]
            device = input_ids.device
            attention_mask = inputs["attention_mask"]
            # (The processor will have prepared video frames as well, typically as pixel values in inputs)

            # Prepare the target token IDs for the labels (class names)
            label_texts = [full_dataset.classes[label] for label in labels]  # convert label indices to class name strings
            # Tokenize each label text (without adding special tokens, we'll add EOS manually)
            target_token_ids = []
            for text in label_texts:
                ids = processor.tokenizer(text, add_special_tokens=False).input_ids
                target_token_ids.append(ids)
            # Get EOS token id (special end-of-sequence token for the model)
            eos_token_id = processor.tokenizer.eos_token_id
            if eos_token_id is None:
                eos_token_id = processor.tokenizer.sep_token_id  # fallback to SEP token if EOS not defined
            for ids in target_token_ids:
                # Append EOS to each target sequence
                if eos_token_id is not None:
                    ids.append(eos_token_id)
            
            # Now we have `input_ids` for the prompt and `target_token_ids` for each label.
            # We need to concatenate them so that model input includes the prompt followed by the label tokens.
            # We'll also construct the labels tensor such that only the label part is used for computing loss.
            batch_max_prompt_len = attention_mask.sum(dim=1).max().item()  # length of longest prompt (text+video) in batch
            combined_input_ids = []
            combined_attention_mask = []
            combined_labels = []
            for i in range(batch_size):
                # Actual length of prompt (non-padded) for sample i
                prompt_len = int(attention_mask[i].sum().item())
                prompt_ids = input_ids[i, :prompt_len]  # slice out the prompt tokens (text tokens; video handled separately)
                # Get target (label) ids and convert to tensor
                target_ids_tensor = torch.tensor(target_token_ids[i], dtype=torch.long, device=device)                # Combine prompt and target
                combined_ids = torch.cat([prompt_ids, target_ids_tensor], dim=0)
                # Create combined attention mask (1s for each real token)
                combined_mask = torch.ones(combined_ids.size(0), dtype=torch.long, device=device)                # Create labels: -100 for prompt part (so loss is not computed on them), and actual ids for target part
                labels_masked = torch.full_like(combined_ids, fill_value=-100, device=device)
                if target_ids_tensor.numel() > 0:
                    labels_masked[prompt_ids.size(0):] = target_ids_tensor  # only predict the target tokens

                combined_input_ids.append(combined_ids)
                combined_attention_mask.append(combined_mask)
                combined_labels.append(labels_masked)
            # Pad the combined sequences in the batch to the same length
            combined_input_ids = torch.nn.utils.rnn.pad_sequence(combined_input_ids, batch_first=True, padding_value=processor.tokenizer.pad_token_id or 0)
            combined_attention_mask = torch.nn.utils.rnn.pad_sequence(combined_attention_mask, batch_first=True, padding_value=0)
            combined_labels = torch.nn.utils.rnn.pad_sequence(combined_labels, batch_first=True, padding_value=-100)
            # Move them to device
            combined_input_ids = combined_input_ids.to(model.device)
            combined_attention_mask = combined_attention_mask.to(model.device)
            combined_labels = combined_labels.to(model.device)

            # Forward pass with our prepared inputs and labels. Include video data if present.
            # The processor might include video pixel values in the inputs (e.g., 'videos' key).
            model_inputs = {
                "input_ids": combined_input_ids,
                "attention_mask": combined_attention_mask,
                "labels": combined_labels
            }
            # If the processor returned video pixel data (the key name could be 'videos' or similar), include it:
            if "videos" in inputs:
                model_inputs["videos"] = inputs["videos"]
            elif "pixel_values" in inputs:
                model_inputs["pixel_values"] = inputs["pixel_values"]
            # (We pass use_audio_in_video flag as False because our synthetic videos have no audio track)
            outputs = model(**model_inputs, use_audio_in_video=False)
            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            global_step += 1
            wandb.log({"train/loss": batch_loss, "epoch": epoch}, step=global_step)

            # Print progress occasionally
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx}: loss = {batch_loss:.4f}")
        
        # Epoch-end logging
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average loss: {avg_loss:.4f}")
        
        # Evaluate on validation set if available
        if val_loader is not None:
            model.eval()
            correct = 0
            total = 0
            # We will generate predictions for each video in val set
            for videos, labels in val_loader:
                batch_size = len(videos)
                text_prompts = [cfg.train.system_prompt] * batch_size
                # Prepare inputs (no labels, we ll use generate)
                inputs = processor(text=text_prompts, videos=videos, return_tensors="pt", padding=True, truncation=True).to(model.device)
                # Generate predictions. Limit new tokens to length of longest class name + 1
                with torch.no_grad():
                    gen_outputs = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=5)
                # Decode predictions
                pred_texts = processor.batch_decode(gen_outputs, skip_special_tokens=True)
                # Compare with true labels
                for pred_text, true_idx in zip(pred_texts, labels):
                    pred_text = pred_text.strip().lower().replace(" ", "").replace("_", "")
                    true_text = full_dataset.classes[true_idx].lower().replace(" ", "").replace("_", "")
                    if pred_text == true_text:
                        correct += 1
                    total += 1
            val_acc = correct / total if total > 0 else 0.0
            print(f"Validation accuracy: {val_acc*100:.2f}%")
            wandb.log({"val/accuracy": val_acc, "epoch": epoch}, step=global_step)
            model.train()  # back to train mode for next epoch
    
    # Training complete, save LoRA adapter weights
    save_dir = cfg.model.lora_weights
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving LoRA adapter weights to '{save_dir}'...")
    model.save_pretrained(save_dir)
    print("Training finished and model saved. You can now run demo.py to test the model on a video.")

    wandb.finish()

if __name__ == "__main__":
    
    main()
