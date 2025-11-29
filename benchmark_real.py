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
    train_loader = get_data(t_conf.block_size, t_conf.batch_size, t_conf.device, split='train')
    val_loader = get_data(t_conf.block_size, t_conf.batch_size, t_conf.device, split='val')
    
    # --- Run 1: AdamW ---
    print("\n=== Training with AdamW ===")
    model_adam = GPT(m_conf)
    # Configure optimizer groups (decay vs no decay)
    optim_groups = model_adam.configure_optimizers(t_conf.weight_decay, t_conf.learning_rate, (t_conf.beta1, t_conf.beta2), t_conf.device)
    optimizer = torch.optim.AdamW(optim_groups, lr=t_conf.learning_rate, betas=(t_conf.beta1, t_conf.beta2))
    
    trainer_adam = Trainer(model_adam, optimizer, train_loader, val_loader, t_conf)
    trainer_adam.train('adamw_run')
    
    # --- Run 2: SF-Muon ---
    print("\n=== Training with SF-Muon ===")
    model_muon = GPT(m_conf) # Fresh init for fairness
    # For Muon, we pass all params; the optimizer handles internal grouping/matrix detection
    # But we should respect weight decay settings.
    # SF-Muon handles weight decay internally.
    optimizer = ScheduleFreeMuon(model_muon.parameters(), lr=0.02, weight_decay=0.01)
    
    trainer_muon = Trainer(model_muon, optimizer, train_loader, val_loader, t_conf)
    trainer_muon.train('sf_muon_run')

if __name__ == '__main__':
    # Enable TF32
    torch.set_float32_matmul_precision('high')
    run_experiment()
