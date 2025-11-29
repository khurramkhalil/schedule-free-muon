import torch
from config import ModelConfig, TrainConfig
from model import GPT
from data import get_data
from trainer import Trainer
from sf_muon import ScheduleFreeMuon
import copy

def run_experiment():
    # Config
    m_conf = ModelConfig()
    t_conf = TrainConfig()
    
    print(f"Device: {t_conf.device}")
    
    # Data
    train_loader = get_data(m_conf.block_size, t_conf.batch_size, t_conf.device, split='train')
    val_loader = get_data(m_conf.block_size, t_conf.batch_size, t_conf.device, split='val')
    
    # --- Run 1: AdamW ---
    # print("\n=== Training with AdamW ===")
    # model_adam = GPT(m_conf)
    # # Configure optimizer groups (decay vs no decay)
    # optim_groups = model_adam.configure_optimizers(t_conf.weight_decay, t_conf.learning_rate, (t_conf.beta1, t_conf.beta2), t_conf.device)
    # optimizer = torch.optim.AdamW(optim_groups, lr=t_conf.learning_rate, betas=(t_conf.beta1, t_conf.beta2))
    
    # trainer_adam = Trainer(model_adam, optimizer, train_loader, val_loader, t_conf)
    # trainer_adam.train('adamw_run')
    
    # --- Run 2: SF-Muon ---
    print("\n=== Training with SF-Muon ===")
    model_muon = GPT(m_conf) # Fresh init for fairness
    
    # Separate params for Muon (Matrices) and AdamW (Vectors + Embeddings)
    # Muon paper typically treats Embeddings as vectors (AdamW) or at least doesn't orthogonalize them
    muon_params = []
    adam_params = []
    
    for name, p in model_muon.named_parameters():
        if p.requires_grad:
            # Embeddings and Vectors -> AdamW
            if p.ndim < 2 or "wte" in name or "wpe" in name or "lm_head" in name:
                adam_params.append(p)
            else:
                # Linear Layers -> Muon
                muon_params.append(p)
                
    # Create groups
    # Note: SF-Muon handles weight decay internally, but we can pass it per group
    optim_groups = [
    optim_groups = [
        {'params': muon_params, 'use_muon': True, 'lr': 0.1, 'weight_decay': 0.01, 'momentum': 0.95},
        {'params': adam_params, 'use_muon': False, 'lr': 0.005, 'weight_decay': 0.0}
    ]
        {'params': adam_params, 'use_muon': False, 'lr': 0.005, 'weight_decay': 0.0}
    ]
    # Use explicit LR for AdamW part instead of config override
    # optim_groups[1]['lr'] = t_conf.learning_rate 
    
    optimizer = ScheduleFreeMuon(optim_groups, warmup_steps=100)
    
    trainer_muon = Trainer(model_muon, optimizer, train_loader, val_loader, t_conf)
    trainer_muon.train('sf_muon_run')

if __name__ == '__main__':
    # Enable TF32
    torch.set_float32_matmul_precision('high')
    run_experiment()
