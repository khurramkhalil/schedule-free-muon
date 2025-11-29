from dataclasses import dataclass
import torch

@dataclass
class ModelConfig:
    vocab_size: int = 50257 # GPT-2 vocab size
    d_model: int = 768      # GPT-2 Small
    n_layer: int = 12
    n_head: int = 12
    dropout: float = 0.1
    bias: bool = False      # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster

@dataclass
class TrainConfig:
    batch_size: int = 12
    block_size: int = 1024  # Context length
    max_steps: int = 1000   # Short run for validation
    learning_rate: float = 6e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    log_interval: int = 10
    eval_interval: int = 100
    eval_iters: int = 200
    warmup_steps: int = 100
