import torch
import wandb
import logging
from generator import SyntheticVideoDataset
from torch.utils.data import DataLoader, random_split
from transformers import Qwen2_5OmniModel, Qwen2_5OmniProcessor, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from qwen_omni_utils import process_vision_info  
import numpy as np
import os
import shutil
from PIL import Image
import numpy as np
import imageio
import hashlib
import torch.nn.functional as F

class SuppressQwenWarnings(logging.Filter):
    def filter(self, record):
        return "System prompt modified" not in record.getMessage()

def evaluate_model(
    model,
    processor,
    val_loader,
    dataset_classes,
    epoch: int,
    global_step: int = None,
    max_new_tokens: int = 10
):
    model.eval()
    correct = 0
    total = 0

    for batch in val_loader:
        batch = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        gen_batch = {k: v for k, v in batch.items() if k not in ["label_ids"]}
        print(f"\n[DEBUG] Prompt to generate:\n{processor.tokenizer.batch_decode(batch['input_ids'], skip_special_tokens=False)[0]}")

        with torch.no_grad():
            outputs = model.generate(
                **gen_batch,
                use_audio_in_video=False,
                max_new_tokens=max_new_tokens
            )
        decoded = processor.batch_decode(outputs, skip_special_tokens=True)
        print(f"[Raw Prediction]: {decoded[0]}")
        predictions = processor.batch_decode(outputs, skip_special_tokens=True)

        true_label_ids = batch.get("label_ids", None)
        if true_label_ids is None:
            raise ValueError("Validation batch is missing `label_ids`. Make sure collator attaches them in inference mode.")

        for pred_raw, true_idx in zip(predictions, true_label_ids):
            if "\nassistant\n" in pred_raw:
                pred_answer = pred_raw.split("\nassistant\n")[-1].strip()
            else:
                pred_answer = pred_raw.strip()

            pred_clean = pred_answer.lower().replace(" ", "").replace("_", "")
            true_clean = dataset_classes[true_idx].lower().replace(" ", "").replace("_", "")
            print(f"Comparing: pred='{pred_clean}' vs true='{true_clean}'")
            print(f"[Predicted] {pred_answer}")
            print(f"[Expected ] {dataset_classes[true_idx]}")

            if pred_clean == true_clean:
                correct += 1
            total += 1

    acc = correct / total if total > 0 else 0.0
    print(f"Validation accuracy: {acc * 100:.2f}%")
    wandb.log({"val/accuracy": acc, "epoch": epoch}, step=global_step)
    model.train()
    return acc

def hash_frames(frames: list[Image.Image]) -> str:
    all_bytes = b"".join([f.tobytes() for f in frames])
    return hashlib.md5(all_bytes).hexdigest()[:12]

def save_video_from_frames(
    frames: list[Image.Image],
    output_dir: str,
    prefix: str = "vid",
    fps: int = 2
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    hash_str = hash_frames(frames)
    video_path = os.path.join(output_dir, f"{prefix}_{hash_str}.mp4")

    writer = imageio.get_writer(video_path, fps=fps, codec="libx264", format="FFMPEG")
    for frame in frames:
        np_frame = np.array(frame.convert("RGB"))
        writer.append_data(np_frame)
    writer.close()

    print(f"✅ Saved video: {video_path}")
    return video_path

def cleanup_tmp_videos(tmp_dir: str):
    if os.path.exists(tmp_dir):
        shutil.rmtree(tmp_dir)
        print(f"Deleted temporary video directory: {tmp_dir}")

class VideoPrefetchWrapper:
    def __init__(self, dataset, tmp_dir="tmp_videos", fps=2, resized_height=56, resized_width=56, use_video_files=False):
        self.dataset = dataset
        self.tmp_dir = tmp_dir
        self.fps = fps
        self.resized_height = resized_height
        self.resized_width = resized_width
        self.use_video_files = use_video_files
        self.generated_paths = []

        os.makedirs(self.tmp_dir, exist_ok=True)

        if hasattr(dataset, "classes"):
            self.classes = dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        video, label = self.dataset[idx]

        if isinstance(video, list):
            if getattr(self, "use_video_files", False):
                video_path = save_video_from_frames(
                    video, self.tmp_dir, prefix=f"sample_{idx}", fps=self.fps
                )
                self.generated_paths.append(video_path)
                return video_path, label
            else:
                return video, label  # pass frames directly

        return video, label  

    def cleanup(self):
        if os.environ.get("KEEP_TMP_VIDEOS", "0") != "1":
            cleanup_tmp_videos(self.tmp_dir)


class QwenChatVideoCollator:
    """
    Collator that:
      - Builds Qwen-style system/user messages
      - Includes a video in the user message
      - Inserts the label as the assistant message
      - Uses Qwen's official `apply_chat_template` + `process_mm_info`
      - Creates `labels` that mask out system+user tokens, leaving only assistant tokens supervised
    """
    def __init__(
        self,
        processor,            
        dataset_classes: list[str],
        system_prompt: str,
        classification_prompt: str = "Please classify this video:",
        eos_token_id: int = None,
        train=True
    ):
        """
        Args:
            processor: Qwen2_5OmniProcessor for text+video tokenization.
            dataset_classes: e.g. ["cat", "dog", "fish"] for index->label mapping.
            system_prompt: text for the system role in each sample.
            classification_prompt: text appended to the user message, e.g. "Please classify this video"
            eos_token_id: optional ID for the EOS token to append after the label.
        """
        self.processor = processor
        self.dataset_classes = dataset_classes
        self.system_prompt = system_prompt
        self.classification_prompt = classification_prompt
        self.eos_token_id = eos_token_id
        self.train = train

    def __call__(self, batch: list[tuple[object, int]]) -> dict:
        """
        batch: list of (video_info, label_id) pairs from your dataset.
               `video_info` could be a path or a list of frames. 
        Returns: a dict with the keys:
          "input_ids", "attention_mask", "labels", plus
          "videos" or "pixel_values" if needed by Qwen.
        """
        # Build a Qwen conversation for each sample
        conversations_batch = []
        for (video_content, label_id) in batch:
            label_text = self.dataset_classes[label_id]

            user_content = [
                {"type": "text", "text": self.classification_prompt},
                {"type": "video", "video": video_content},
            ]
            single_sample_conv = [
                {"role": "system",    "content": self.system_prompt},
                {"role": "user",      "content": user_content},
            ]
            # The "assistant" role is the ground truth label
            if self.train:                                
                single_sample_conv.append({"role": "assistant", "content": label_text})

            conversations_batch.append(single_sample_conv)

        #Convert to raw text prompts. We'll do teacher forcing, so no generation prompt:
        prompts = self.processor.apply_chat_template(
            conversations_batch,
            tokenize=False,
            #we add generation prompt in evaluation mode 
            add_generation_prompt=not self.train,
        )

        #Parse the actual video frames out via qwen official utils (process_mm_info)
        images, videos = process_vision_info(
            conversations_batch,
            return_video_kwargs=False
        )

        #Tokenize everything using Qwen’s processor
        inputs = self.processor(
            text=prompts,
            images=images,
            videos=videos,
            return_tensors="pt",
            padding=True,
            truncation=True,
            use_audio_in_video=False
        )

        #everything is -100 except the last part that corresponds to the label text.
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # For each sample, figure out how many tokens correspond to the label text
        if self.train:
            labels = torch.full_like(input_ids, fill_value=-100)
            for i, (video_content, label_id) in enumerate(batch):
                # Re-tokenize the label
                label_text = self.dataset_classes[label_id]
                label_ids = self.processor.tokenizer(label_text, add_special_tokens=False).input_ids
                if self.eos_token_id is not None:
                    label_ids.append(self.eos_token_id)
                label_len = len(label_ids)

                # how many actual tokens are in sample i (i.e. sum of attention_mask)
                real_length = attention_mask[i].sum().item()
                start_idx = real_length - label_len
                if start_idx < 0:
                    # The label is bigger than we expected or the prompt truncated more than planned
                    start_idx = 0  # clamp to avoid negative slices

                # Now copy those tokens from input_ids into labels
                labels[i, start_idx:real_length] = input_ids[i, start_idx:real_length]

            inputs["labels"] = labels
        inputs["label_ids"] = torch.tensor([label_id for (_, label_id) in batch])
        return inputs  


def get_data_loaders(cfg):
    full_dataset = SyntheticVideoDataset(num_videos=cfg.dataset.num_videos,
                                        num_classes=cfg.dataset.num_classes,
                                        frames_per_video=cfg.dataset.frames_per_video,
                                        frame_height=cfg.dataset.frame_height,
                                        frame_width=cfg.dataset.frame_width,
                                        max_skip=cfg.dataset.max_skip,
                                        seed=cfg.dataset.seed)
    # split into training and validation sets (80/20 split here)
    train_size = int((max(0,1-cfg.train.val_split)) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    if val_size > 0:
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], 
                                                  generator=torch.Generator().manual_seed(cfg.dataset.seed or 42))
    else:
        train_dataset = full_dataset
        val_dataset = None
    train_dataset.classes = full_dataset.classes
    if val_dataset:
        val_dataset.classes = full_dataset.classes
    # Create DataLoader 

    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.model.base_model)

    classification_prompt = (
    "Please classify this video. Choose one of: "
    + ", ".join(train_dataset.classes )
    )
    train_dataset = VideoPrefetchWrapper(
        train_dataset,
        tmp_dir=cfg.dataset.temp_folder,
        fps=24,
        resized_height=cfg.dataset.frame_height,
        resized_width=cfg.dataset.frame_width,
        use_video_files=cfg.dataset.use_video_files,  
    )

    if val_dataset is not None:
        val_dataset = VideoPrefetchWrapper(
            val_dataset,
            tmp_dir=cfg.dataset.temp_folder,
            fps=2,
            resized_height=cfg.dataset.frame_height,
            resized_width=cfg.dataset.frame_width,
            use_video_files=cfg.dataset.use_video_files,  
        )

    collator_train = QwenChatVideoCollator(
        processor=processor,
        dataset_classes=train_dataset.classes,
        system_prompt=cfg.train.system_prompt,
        classification_prompt=classification_prompt,
        eos_token_id=processor.tokenizer.eos_token_id
    )

    collator_val = QwenChatVideoCollator(
        processor=processor,
        dataset_classes=train_dataset.classes,
        system_prompt=cfg.train.system_prompt,
        classification_prompt=classification_prompt,
        eos_token_id=processor.tokenizer.eos_token_id,
        train=False
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True, collate_fn=collator_train)
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=False, collate_fn=collator_val)

    return train_loader, val_loader

def get_model_and_processor(cfg):
    model_name = cfg.model.base_model
    print(f"Loading base model '{model_name}'...")
    
    if cfg.model.use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = Qwen2_5OmniModel.from_pretrained(model_name, device_map="auto", quantization_config=bnb_config)
        model = model.thinker
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[cfg.train.precision]
        model = Qwen2_5OmniModel.from_pretrained(model_name, device_map="auto", torch_dtype=dtype, attn_implementation="flash_attention_2")
        model = model.thinker

    model.train()
  
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    lora_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=list(cfg.model.lora_target_modules), 
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    return model, processor

def print_model_params(model):
    # Print number of trainable parameters 
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Model loaded. Total parameters: {total_params}, Trainable (LoRA) parameters: {trainable_params}")


def contrastive_loss(embeddings, labels, temperature=0.07):
    normed = F.normalize(embeddings, dim=-1)  # (B, D)
    logits = normed @ normed.T  # cosine similarity matrix (B, B)
    logits /= temperature

    labels = labels.unsqueeze(0) == labels.unsqueeze(1)  # (B, B) similarity mask
    labels = labels.float()

    # Mask out self-comparisons
    mask = torch.eye(labels.size(0), device=labels.device)
    labels = labels * (1 - mask)

    # Contrastive loss
    log_probs = F.log_softmax(logits, dim=1)
    loss = -(labels * log_probs).sum(dim=1) / labels.sum(dim=1).clamp(min=1e-6)
    return loss.mean()