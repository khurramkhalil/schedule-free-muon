import time
import torch
import torch.nn.functional as F
from collections import defaultdict
import csv
import os

class Trainer:
    def __init__(self, model, optimizer, train_loader, val_loader, config):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        
    def train(self, run_name):
        print(f"Starting training run: {run_name}")
        self.model.train()
        
        # Logging setup
        log_dir = 'logs'
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f'{run_name}.csv')
        
        with open(log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'time', 'train_loss', 'val_loss', 'tokens_per_sec'])
            
        iter_time = time.time()
        tokens_processed = 0
        t0 = time.time()
        
        step = 0
        data_iter = iter(self.train_loader)
        
        while step < self.config.max_steps:
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(self.train_loader)
                x, y = next(data_iter)
                
            x, y = x.to(self.device), y.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if self.device == 'cuda' else torch.no_grad():
                logits, loss = self.model(x, y)
            
            # Backward pass
            loss.backward()
            
            if self.config.grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                
            self.optimizer.step()
            
            # Logging
            t1 = time.time()
            dt = t1 - iter_time
            iter_time = t1
            tokens_processed += x.numel()
            
            if step % self.config.log_interval == 0:
                tokens_per_sec = x.numel() / dt
                print(f"step {step}: loss {loss.item():.4f}, time {dt*1000:.2f}ms, {tokens_per_sec:.2f} tok/s")
                
                with open(log_file, 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([step, t1 - t0, loss.item(), '', tokens_per_sec])
                    
            if step % self.config.eval_interval == 0:
                val_loss = self.evaluate()
                print(f"step {step}: val loss {val_loss:.4f}")
                # Update log with val loss
                # (Simple append for now, ideally would align rows)
                
            step += 1
            
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        # Schedule-Free Optimizer support: switch to averaged weights
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()
            
        losses = []
        for i, (x, y) in enumerate(self.val_loader):
            if i >= self.config.eval_iters: break
            x, y = x.to(self.device), y.to(self.device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16) if self.device == 'cuda' else torch.no_grad():
                logits, loss = self.model(x, y)
            losses.append(loss.item())
            
        # Schedule-Free Optimizer support: switch back to training weights
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
            
        self.model.train()
        return sum(losses) / len(losses)
