import os
import json
import math
import argparse
import csv
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from model import GPTLanguageModel
from tokenizer_wrapper import TokenizerWrapper
from dataset import get_dataloader

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to model config json")
    parser.add_argument("--tokenizer", type=str, required=True, help="Path to tokenizer json")
    parser.add_argument("--dataset", type=str, required=True, help="HuggingFace dataset path")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--max_steps", type=int, default=50000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=5000)
    parser.add_argument("--val_steps", type=int, default=1000)
    parser.add_argument("--val_batches", type=int, default=100)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from (or 'latest')")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load config and tokenizer
    with open(args.config, "r") as f:
        config = json.load(f)
    tokenizer = TokenizerWrapper(args.tokenizer)
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model setup
    model = GPTLanguageModel(**config).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"🚀 Using {torch.cuda.device_count()} GPUs with DataParallel!")
        model = torch.nn.DataParallel(model)
        
    # Function to get inner model state dict (handles DataParallel)
    def get_model_state_dict(m):
        return m.module.state_dict() if hasattr(m, 'module') else m.state_dict()

    print(f"Total parameters: {sum(p.numel() for p in (model.module.parameters() if hasattr(model, 'module') else model.parameters())) / 1e6:.2f} M")
    
    # Optimizer & Scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1, betas=(0.9, 0.95))
    scheduler = get_cosine_schedule_with_warmup(optimizer, args.warmup_steps, args.max_steps)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))
    
    start_step = 0

    # Auto-resume from latest checkpoint if requested or present
    ckpt_path = args.resume
    if ckpt_path == "latest" or (ckpt_path is None and os.path.exists(os.path.join(args.output_dir, "latest.pt"))):
        ckpt_path = os.path.join(args.output_dir, "latest.pt")
    
    if ckpt_path and os.path.exists(ckpt_path):
        print(f"🔄 Resuming training from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        if hasattr(model, 'module'):
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_step = checkpoint['step']
        print(f"✅ Resumed at step {start_step}")
    
    # Data loaders
    train_loader = get_dataloader(args.dataset, tokenizer, config["context_len"], split="train", batch_size=args.batch_size)
    val_loader = get_dataloader(args.dataset, tokenizer, config["context_len"], split="validation", batch_size=args.batch_size)

    train_iter = iter(train_loader)

    # CSV log file for loss curves
    log_path = os.path.join(args.output_dir, "train_log.csv")
    log_file = open(log_path, "a" if start_step > 0 else "w", newline="")
    log_writer = csv.writer(log_file)
    if start_step == 0:
        log_writer.writerow(["step", "train_loss", "val_loss", "val_ppl", "lr"])
    
    # Training Loop
    step = start_step
    pbar = tqdm(total=args.max_steps, initial=start_step, desc="Training")
    
    while step < args.max_steps:
        model.train()
        optimizer.zero_grad()
        loss_accum = 0.0
        
        for _ in range(args.grad_accum):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
                
            x, y = x.to(device), y.to(device)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                logits, loss = model(x, targets=y)
                # If DataParallel, loss is a vector (one per GPU), take the mean
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / args.grad_accum
                
            scaler.scale(loss).backward()
            loss_accum += loss.item()
            
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        step += 1
        pbar.update(1)
        
        if step % 10 == 0:
            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({'loss': f"{loss_accum:.4f}", 'lr': f"{current_lr:.2e}"})
            log_writer.writerow([step, f"{loss_accum:.6f}", "", "", f"{current_lr:.6e}"])
            log_file.flush()
            
        # Validation
        if step % args.val_steps == 0:
            model.eval()
            val_loss = 0.0
            val_iter = iter(val_loader)
            with torch.no_grad():
                for i in range(args.val_batches):
                    try:
                        vx, vy = next(val_iter)
                    except StopIteration:
                        break
                    vx, vy = vx.to(device), vy.to(device)
                    with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                        _, vl = model(vx, targets=vy)
                        if vl.dim() > 0:
                            vl = vl.mean()
                    val_loss += vl.item()
            val_loss /= min(args.val_batches, i + 1)
            val_ppl = math.exp(min(val_loss, 20))
            print(f"\nStep {step}: Val Loss {val_loss:.4f} | Val PPL {val_ppl:.2f}")
            log_writer.writerow([step, "", f"{val_loss:.6f}", f"{val_ppl:.4f}", ""])
            log_file.flush()
            
        # Save Checkpoint
        if step % args.save_steps == 0:
            checkpoint = {
                'step': step,
                'model_state_dict': get_model_state_dict(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'config': config
            }
            torch.save(checkpoint, os.path.join(args.output_dir, f"ckpt_{step}.pt"))
            torch.save(checkpoint, os.path.join(args.output_dir, "latest.pt"))
            
    pbar.close()
    log_file.close()
    
    # Save final model
    torch.save({
        'step': step,
        'model_state_dict': get_model_state_dict(model),
        'config': config
    }, os.path.join(args.output_dir, "model_final.pt"))
    print("Training complete!")

if __name__ == "__main__":
    train()
