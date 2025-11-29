import torch
from config import ModelConfig, TrainConfig
from model import GPT
from data import get_data
from trainer import Trainer
from sf_muon import ScheduleFreeMuon

def run_experiment():
    # Config
    m_conf = ModelConfig()
    t_conf = TrainConfig()
    
    print(f"Device: {t_conf.device}")
    
    # Data (load once, reuse for both)
    train_loader = get_data(m_conf.block_size, t_conf.batch_size, t_conf.device, split='train')
    val_loader = get_data(m_conf.block_size, t_conf.batch_size, t_conf.device, split='val')
    
    # # --- Run 1: AdamW ---
    # print("\n=== Training with AdamW ===")
    # torch.manual_seed(42)  # Set seed
    # model_adam = GPT(m_conf)
    # optim_groups = model_adam.configure_optimizers(
    #     t_conf.weight_decay, t_conf.learning_rate, 
    #     (t_conf.beta1, t_conf.beta2), t_conf.device
    # )
    # optimizer_adam = torch.optim.AdamW(
    #     optim_groups, lr=t_conf.learning_rate, 
    #     betas=(t_conf.beta1, t_conf.beta2)
    # )
    
    # trainer_adam = Trainer(model_adam, optimizer_adam, train_loader, val_loader, t_conf)
    # trainer_adam.train('adamw_run')
    
    # --- Run 2: SF-Muon ---
    print("\n=== Training with SF-Muon ===")
    torch.manual_seed(42)  # SAME seed for fair comparison
    model_muon = GPT(m_conf)
    
    # Use ALL parameters with single LR (optimizer handles routing internally)
    optimizer_muon = ScheduleFreeMuon(
        model_muon.parameters(),
        lr=t_conf.learning_rate,  # 6e-4
        weight_decay=t_conf.weight_decay,
        warmup_steps=100
    )
    
    trainer_muon = Trainer(model_muon, optimizer_muon, train_loader, val_loader, t_conf)
    trainer_muon.train('sf_muon_run')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    run_experiment()