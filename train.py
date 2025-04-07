import os
import torch
import logging
import wandb
import hydra
from omegaconf import OmegaConf, DictConfig
from utils import SuppressQwenWarnings, evaluate_model, get_data_loaders, get_model_and_processor, print_model_params


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Pretty-print the config for verification
    print("Configuration:\n", OmegaConf.to_yaml(cfg))
    os.chdir(hydra.utils.get_original_cwd() or ".")

    # Set up Weights & Biases logging
    wandb.init(project=cfg.wandb.project, name=cfg.wandb.run_name, mode=cfg.wandb.mode,
               config=OmegaConf.to_container(cfg, resolve=True))

    train_loader, val_loader = get_data_loaders(cfg)
    # Load the Qwen2.5-Omni model and processor
    model, processor = get_model_and_processor(cfg)

    if cfg.model.print_parameters:
        print_model_params(model)

    # Set up optimizer (only parameters with requires_grad=True, which will be the LoRA parameters)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg.train.learning_rate)

    # Training loop
    global_step = 0
    for epoch in range(1, cfg.train.num_epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(train_loader):
            batch_dict = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()}
            outputs = model(**batch_dict, use_audio_in_video=False)
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
            if step % 10 == 0:
                print(f"Epoch {epoch} Batch {step}: loss = {batch_loss:.4f}")
            # Do eval after steps
            if step % cfg.train.eval_after_steps == 0:
                if val_loader is not None:
                    val_acc = evaluate_model(
                            model=model,
                            val_loader=val_loader,
                            processor=processor,
                            dataset_classes=val_loader.dataset.classes,
                            epoch=epoch,
                            global_step=global_step
                        )
                    print(f"Eval after {step} steps: Acc = {val_acc:.4f}")

        
        # Epoch-end logging
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} finished. Average loss: {avg_loss:.4f}")
        
        # # Evaluate on validation set if available
        model.train()

        if val_loader is not None:
            val_acc = evaluate_model(
                model=model,
                val_loader=val_loader,
                processor=processor,
                dataset_classes=val_loader.dataset.classes,
                epoch=epoch,
                global_step=global_step
            )


    
    # Training complete, save LoRA adapter weights
    save_dir = cfg.model.lora_weights
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving LoRA adapter weights to '{save_dir}'...")
    model.save_pretrained(save_dir)
    print("Training finished and model saved. You can now run demo.py to test the model on a video.")
    if cfg.dataset.delete_videos:
        train_loader.dataset.cleanup()
        val_loader.dataset.cleanup()
        print("Temporary videos cleanup done.")

    wandb.finish()

if __name__ == "__main__":
    logging.getLogger("root").addFilter(SuppressQwenWarnings())
    main()
