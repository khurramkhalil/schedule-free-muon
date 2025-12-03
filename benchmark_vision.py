import torch
from config import TrainConfig
from model_vision import ResNet18, ViT
from data_vision import get_cifar10_data
from trainer import Trainer
from sf_muon import ScheduleFreeMuon
import os

def run_vision_experiment():
    # Vision Config
    # Override defaults for CIFAR-10
    t_conf = TrainConfig()
    t_conf.batch_size = 128
    t_conf.max_steps = 2000 # ~5 epochs
    t_conf.learning_rate = 1e-3 # Standard AdamW LR for ResNet
    t_conf.weight_decay = 0.05
    t_conf.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    t_conf.log_interval = 50
    t_conf.eval_interval = 200
    t_conf.eval_iters = 50 # Evaluate on 50 batches
    
    print(f"Device: {t_conf.device}")
    
    # Data
    train_loader, val_loader = get_cifar10_data(t_conf.batch_size)
    
    # ========================================================================
    # Experiment 1: ResNet-18
    # ========================================================================
    print("\n" + "="*50)
    print("Experiment 1: ResNet-18 on CIFAR-10")
    print("="*50)
    
    # --- Run 1a: AdamW ---
    print("\n--- Training ResNet-18 with AdamW ---")
    torch.manual_seed(42)
    model_adam = ResNet18()
    
    # Configure optim groups
    optim_groups = model_adam.configure_optimizers(
        t_conf.weight_decay, t_conf.learning_rate, 
        (t_conf.beta1, t_conf.beta2), t_conf.device
    )
    optimizer_adam = torch.optim.AdamW(
        optim_groups, lr=t_conf.learning_rate, 
        betas=(t_conf.beta1, t_conf.beta2)
    )
    
    trainer_adam = Trainer(model_adam, optimizer_adam, train_loader, val_loader, t_conf)
    trainer_adam.train('vision_resnet_adamw')
    
    # --- Run 1b: SF-Muon ---
    print("\n--- Training ResNet-18 with SF-Muon ---")
    torch.manual_seed(42)
    model_muon = ResNet18()
    
    # Use ALL parameters with single LR (optimizer handles routing internally)
    # Note: Muon LR usually needs to be higher (0.02)
    optimizer_muon = ScheduleFreeMuon(
        model_muon.parameters(),
        lr=0.02, 
        weight_decay=t_conf.weight_decay,
        warmup_steps=200
    )
    
    trainer_muon = Trainer(model_muon, optimizer_muon, train_loader, val_loader, t_conf)
    trainer_muon.train('vision_resnet_muon')

    # ========================================================================
    # Experiment 2: Vision Transformer (ViT)
    # ========================================================================
    print("\n" + "="*50)
    print("Experiment 2: ViT (Small) on CIFAR-10")
    print("="*50)
    
    # --- Run 2a: AdamW ---
    print("\n--- Training ViT with AdamW ---")
    torch.manual_seed(42)
    model_vit_adam = ViT()
    
    optim_groups = model_vit_adam.configure_optimizers(
        t_conf.weight_decay, t_conf.learning_rate, 
        (t_conf.beta1, t_conf.beta2), t_conf.device
    )
    optimizer_vit_adam = torch.optim.AdamW(
        optim_groups, lr=t_conf.learning_rate, 
        betas=(t_conf.beta1, t_conf.beta2)
    )
    
    trainer_vit_adam = Trainer(model_vit_adam, optimizer_vit_adam, train_loader, val_loader, t_conf)
    trainer_vit_adam.train('vision_vit_adamw')
    
    # --- Run 2b: SF-Muon ---
    print("\n--- Training ViT with SF-Muon ---")
    torch.manual_seed(42)
    model_vit_muon = ViT()
    
    optimizer_vit_muon = ScheduleFreeMuon(
        model_vit_muon.parameters(),
        lr=0.02,
        weight_decay=t_conf.weight_decay,
        warmup_steps=200
    )
    
    trainer_vit_muon = Trainer(model_vit_muon, optimizer_vit_muon, train_loader, val_loader, t_conf)
    trainer_vit_muon.train('vision_vit_muon')

if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')
    run_vision_experiment()
