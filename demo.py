import os
import torch
import hydra
from omegaconf import DictConfig
from peft import PeftModel
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniModel, BitsAndBytesConfig
from qwen_omni_utils import process_mm_info  # Ensure this is accessible or adjust import
from generator import SyntheticVideoDataset
from PIL import Image
import cv2


def load_video_as_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def run_inference(model, processor, sys_prompt, prompt_text, video_frames, device, max_new_tokens=50):
    messages = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": [
            {"type": "text", "text": prompt_text},
            {"type": "video", "video": video_frames},
        ]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    audios, images, videos = process_mm_info(messages, use_audio_in_video=False)

    inputs = processor(
        text=text,
        audios=audios,
        images=images,
        videos=videos,
        return_tensors="pt",
        padding=True,
        truncation=True,
        use_audio_in_video=False
    )
    inputs = inputs.to(device).to(model.dtype)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            use_audio_in_video=False,
            return_audio=False,
            max_new_tokens=max_new_tokens
        )

    decoded = processor.batch_decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return decoded[0].strip()


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd() or ".")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base + LoRA model
    bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = Qwen2_5OmniModel.from_pretrained(cfg.model.base_model, device_map="auto", quantization_config=bnb_config)
    model = model.to(device)
    model.thinker = PeftModel.from_pretrained(model.thinker, cfg.demo.lora_weights)
    model.eval()

    processor = Qwen2_5OmniProcessor.from_pretrained(cfg.model.base_model)
    sys_prompt = cfg.train.system_prompt
    prompt_text = "Please classify this video. Choose one of: horizontal_move, vertical_move, blinking_dot, random_teleport, bouncing_diag. Respond with only the label."

    if cfg.demo.use_synthetic_dataset:
        dataset = SyntheticVideoDataset(
            num_videos=cfg.dataset.num_videos,
            num_classes=cfg.dataset.num_classes,
            frames_per_video=cfg.dataset.frames_per_video,
            frame_height=cfg.dataset.frame_height,
            frame_width=cfg.dataset.frame_width,
            max_skip=cfg.dataset.max_skip,
            seed=cfg.dataset.seed
        )
        for i in range(min(cfg.demo.num_samples, len(dataset))):
            video_frames, label_id = dataset[i]
            result = run_inference(
                model, processor, sys_prompt, prompt_text, video_frames,
                device, max_new_tokens=cfg.demo.max_new_tokens
            )
            print(f"\nSample {i+1}: Label = {dataset.classes[label_id]}")
            print("Model prediction:", result)
    else:
        video_path = cfg.demo.video_path
        if not os.path.isfile(video_path):
            print(f"Invalid video path: {video_path}")
            return
        video_frames = load_video_as_frames(video_path)
        if len(video_frames) == 0:
            print("No frames extracted from video.")
            return
        result = run_inference(
            model, processor, sys_prompt, prompt_text, video_frames,
            device, max_new_tokens=cfg.demo.max_new_tokens
        )
        print("\nModel prediction:", result)

if __name__ == "__main__":
    main()
