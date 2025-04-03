import os
import random
import torch
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig
from torch.utils.data import random_split
from transformers import (
    Qwen2_5OmniModel, 
    Qwen2_5OmniProcessor,
    TrainingArguments,
    Trainer
)
from peft import LoraConfig, get_peft_model
from dataset import SyntheticVideoDataset

# ------------------------
# 1) Define a DataCollator that performs the same logic
#    you were doing in your training loop for each batch.
# ------------------------
class VideoDataCollator:
    def __init__(self, processor, system_prompt, dataset_classes, eos_token_id=None):
        """
        processor: Qwen2_5OmniProcessor
        system_prompt: str (the base text used for each sample)
        dataset_classes: e.g. full_dataset.classes to map label -> label_name
        eos_token_id: end-of-sequence token
        """
        self.processor = processor
        self.system_prompt = system_prompt
        self.dataset_classes = dataset_classes
        self.eos_token_id = eos_token_id

    def __call__(self, batch):
        """
        batch is a list of items, each item is (frames, label_id) from SyntheticVideoDataset.
        We'll build the final 'input_ids', 'attention_mask', 'labels', etc. for the model.
        """
        # Separate out videos & label IDs
        videos = [item[0] for item in batch]
        label_ids = [item[1] for item in batch]
        
        # Build text prompts from system_prompt
        text_prompts = [self.system_prompt] * len(videos)

        # Preprocess with the Qwen2.5-Omni processor
        inputs = self.processor(
            text=text_prompts,
            videos=videos,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        
        # Convert label_ids to label text (e.g. "class_3"), then tokenize
        label_texts = [self.dataset_classes[lid] for lid in label_ids]
        target_token_ids = []
        for txt in label_texts:
            # e.g. remove special tokens, then add EOS if it exists
            ids = self.processor.tokenizer(txt, add_special_tokens=False).input_ids
            if self.eos_token_id is not None:
                ids.append(self.eos_token_id)
            target_token_ids.append(ids)

        # Now do the same "concatenate prompt + label" logic
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        batch_size = input_ids.shape[0]
        
        combined_input_ids = []
        combined_attention_mask = []
        combined_labels = []
        
        for i in range(batch_size):
            # length of the non-padded prompt
            prompt_len = attention_mask[i].sum().item()
            prompt_ids = input_ids[i, :prompt_len]
            
            label_ids_tensor = torch.tensor(target_token_ids[i], dtype=torch.long)
            # Merge
            combo_ids = torch.cat([prompt_ids, label_ids_tensor], dim=0)
            
            combo_mask = torch.ones_like(combo_ids)
            labels_masked = torch.full_like(combo_ids, -100)
            
            if label_ids_tensor.numel() > 0:
                labels_masked[prompt_ids.size(0):] = label_ids_tensor
            
            combined_input_ids.append(combo_ids)
            combined_attention_mask.append(combo_mask)
            combined_labels.append(labels_masked)
        
        # Pad
        padded_input_ids = torch.nn.utils.rnn.pad_sequence(
            combined_input_ids,
            batch_first=True,
            padding_value=self.processor.tokenizer.pad_token_id or 0
        )
        padded_attention_mask = torch.nn.utils.rnn.pad_sequence(
            combined_attention_mask,
            batch_first=True,
            padding_value=0
        )
        padded_labels = torch.nn.utils.rnn.pad_sequence(
            combined_labels,
            batch_first=True,
            padding_value=-100
        )
        
        # Build final batch dictionary
        batch_dict = {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "labels": padded_labels
        }
        
        # If the processor returned pixel data, attach that as well
        if "videos" in inputs:
            batch_dict["videos"] = inputs["videos"]
        elif "pixel_values" in inputs:
            batch_dict["pixel_values"] = inputs["pixel_values"]
        
        return batch_dict


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Step 1: Basic Hydra & W&B setup
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    os.chdir(hydra.utils.get_original_cwd() or ".")
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.run_name,
        mode=cfg.wandb.mode,
        config=OmegaConf.to_container(cfg, resolve=True)
    )
    
    # Step 2: Create dataset + train/val split
    full_dataset = SyntheticVideoDataset(
        num_videos=cfg.dataset.num_videos,
        num_classes=cfg.dataset.num_classes,
        frames_per_video=cfg.dataset.frames_per_video,
        frame_height=cfg.dataset.frame_height,
        frame_width=cfg.dataset.frame_width,
        max_skip=cfg.dataset.max_skip,
        seed=cfg.dataset.seed
    )
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(cfg.dataset.seed or 42)
        )
    else:
        train_dataset = full_dataset
        val_dataset = None

    # Step 3: Load Qwen2.5-Omni base, then select .thinker
    model_name = cfg.model.base_model
    print(f"Loading base model '{model_name}'...")
    if cfg.model.use_8bit:
        model = Qwen2_5OmniModel.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
            torch_dtype=torch.float16
        )
    else:
        if cfg.train.precision == "fp16":
            dtype = torch.float16
        elif cfg.train.precision == "bf16":
            dtype = torch.bfloat16
        else:
            dtype = torch.float32
        model = Qwen2_5OmniModel.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=dtype
        )
    # Keep only the text+video submodule
    model = model.thinker
    model.train()
    
    # Step 4: LoRA wrapping
    lora_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=list(cfg.model.lora_target_modules),
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.train()

    # Step 5: Load processor & define data collator
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
    # We'll add an EOS token if present
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is None:
        eos_token_id = processor.tokenizer.sep_token_id

    data_collator = VideoDataCollator(
        processor=processor,
        system_prompt=cfg.train.system_prompt,
        dataset_classes=full_dataset.classes,
        eos_token_id=eos_token_id
    )

    # Step 6: Use HF's Trainer
    # We'll define normal TrainingArguments. For W&B, set `report_to=["wandb"]`.
    from transformers import TrainingArguments, Trainer
    
    training_args = TrainingArguments(
        output_dir=cfg.model.lora_weights,
        overwrite_output_dir=True,
        num_train_epochs=cfg.train.num_epochs,
        per_device_train_batch_size=cfg.train.batch_size,
        per_device_eval_batch_size=cfg.train.batch_size,
        logging_steps=10,
        evaluation_strategy="epoch" if val_dataset is not None else "no",
        save_strategy="no",  # or "epoch" if you want partial saves
        report_to=["wandb"],
        learning_rate=cfg.train.learning_rate,
        fp16=(cfg.train.precision == "fp16"),
        bf16=(cfg.train.precision == "bf16"),
        run_name=cfg.wandb.run_name,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )

    # Step 7: Train
    trainer.train()
    # Optionally evaluate or do trainer.evaluate() if needed
    
    # Step 8: Save the final LoRA adapter
    print(f"Saving LoRA adapter weights to {cfg.model.lora_weights} ...")
    model.save_pretrained(cfg.model.lora_weights)
    print("Training finished. LoRA saved. You can run demo.py to test the model on a video.")

    wandb.finish()

if __name__ == "__main__":
    main()
