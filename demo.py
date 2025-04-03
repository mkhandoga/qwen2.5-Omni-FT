import os
import torch
import hydra
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, Qwen2_5OmniProcessor
from peft import PeftModel
import cv2
from PIL import Image

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Ensure working directory is project root (in case Hydra changes it)
    os.chdir(hydra.utils.get_original_cwd() or ".")
    video_path = cfg.demo.video_path
    if video_path == "" or not os.path.isfile(video_path):
        print("Please specify a valid video file path via demo.video_path")
        return

    # Determine device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model
    model_name = cfg.demo.base_model
    print(f"Loading base model '{model_name}' for inference...")
    if cfg.model.use_8bit and torch.cuda.is_available():
        # Use 8-bit loading if available (only works on GPU)
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", load_in_8bit=True, torch_dtype=torch.float16)
    else:
        base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    # Load LoRA adapter weights
    adapter_dir = cfg.demo.lora_weights
    print(f"Loading LoRA adapter from '{adapter_dir}'...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model = model.to(device)
    model.eval()

    # Load processor for text and video
    processor = Qwen2_5OmniProcessor.from_pretrained(model_name)

    # Read video frames using OpenCV
    cap = cv2.VideoCapture(video_path)
    frames = []
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        # Convert BGR (OpenCV) to RGB and then to PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    if len(frames) == 0:
        print(f"Failed to read any frames from video: {video_path}")
        return

    # Prepare input for the model (system prompt text + video frames)
    sys_prompt = cfg.train.system_prompt  # using the same prompt used in training
    inputs = processor(text=[sys_prompt], videos=[frames], return_tensors="pt", padding=True, truncation=True)
    inputs = inputs.to(device)
    # Generate output (predicted label). Limit generation length to avoid rambling.
    with torch.no_grad():
        output_ids = model.generate(**inputs, use_audio_in_video=False, max_new_tokens=cfg.demo.max_new_tokens)
    # Decode the generated tokens to text
    result_text = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"\nPredicted label: {result_text}")

if __name__ == "__main__":
    main()
