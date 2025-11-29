import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import statistics
import math
import copy
from sf_muon import ScheduleFreeMuon
from stress_test_manifold import run_stress_test

def test_manifold(args):
    """Wrapper for the geometric stress test."""
    print("Running Manifold Recovery Test...")
    success = run_stress_test(dim=1024, steps=300, device=args.device)
    if success:
        print("✅ Manifold Test Passed")
        return True
    else:
        print("❌ Manifold Test Failed")
        return False

# Enable TF32 for faster matrix multiplications on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

def run_benchmark(name, optimizer, model, data, steps=50, warmup=10):
    device = data.device
    
    # Warmup
    print(f"  Warmup ({warmup} steps)...")
    for _ in range(warmup):
        optimizer.zero_grad()
        out = model(data)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    
    times = []
    start_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
    
    print(f"  Benchmarking ({steps} steps)...")
    for _ in range(steps):
        optimizer.zero_grad()
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
            start_event.record()
            start_cpu = time.perf_counter()
        else:
            start_cpu = time.perf_counter()
            
        out = model(data)
        loss = out.mean()
        loss.backward()
        optimizer.step()
        
        if device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            elapsed = start_event.elapsed_time(end_event) / 1000.0 # ms to seconds
        else:
            elapsed = time.perf_counter() - start_cpu
            
        times.append(elapsed)
        
    max_memory = torch.cuda.max_memory_allocated() / (1024**2) if device.type == 'cuda' else 0
    
    avg_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    throughput = (data.size(0) * data.size(1)) / avg_time # tokens/sec
    
    return {
        'name': name,
        'avg_time': avg_time,
        'std_time': std_time,
        'throughput': throughput,
        'memory': max_memory
    }

def test_cost(args):
    """Benchmark computational cost vs AdamW."""
    print("\n" + "="*60)
    print(" COMPUTATIONAL COST BENCHMARK (Publication Quality)")
    print("="*60)
    
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("⚠️ CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    
    print(f"Device: {device}")
    
    # Setup: 6-layer Transformer Encoder
    d_model = 1024 # Increased for more realistic load
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
        num_layers=6
    ).to(device)
    
    batch_size = 32
    seq_len = 128
    data = torch.randn(batch_size, seq_len, d_model, device=device)
    
    optimizers = [
        ('AdamW', lambda p: torch.optim.AdamW(p, lr=1e-3)),
        ('SF-Muon', lambda p: ScheduleFreeMuon(p, lr=0.02))
    ]
    
    results = []
    
    for name, opt_fn in optimizers:
        print(f"\nBenchmarking {name}...")
        # Re-init model/optimizer to ensure fair comparison
        model_copy = copy.deepcopy(model)
        opt = opt_fn(model_copy.parameters())
        
        res = run_benchmark(name, opt, model_copy, data)
        results.append(res)
        
        print(f"  Time: {res['avg_time']*1000:.2f} ± {res['std_time']*1000:.2f} ms/step")
        print(f"  Throughput: {res['throughput']:.0f} tokens/sec")
        if device.type == 'cuda':
            print(f"  Peak Memory: {res['memory']:.2f} MB")
    
    print("\n" + "="*60)
    print(f"{'Optimizer':<15} {'Time (ms)':<15} {'Throughput':<15} {'Slowdown':<10}")
    print("-" * 60)
    
    base_time = results[0]['avg_time']
    for res in results:
        slowdown = res['avg_time'] / base_time
        print(f"{res['name']:<15} {res['avg_time']*1000:<15.2f} {res['throughput']:<15.0f} {slowdown:<10.2f}x")
    print("="*60 + "\n")

def test_convergence(args):
    """Simple convergence test on a synthetic task."""
    print("\n" + "="*60)
    print(" CONVERGENCE TEST (Synthetic Regression)")
    print("="*60)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
        
    dim = 256
    # Linear Regression with Orthogonal Target (Muon-compatible)
    model = nn.Linear(dim, dim, bias=False).to(device)
    
    # Initialize model close to identity to help convergence
    nn.init.orthogonal_(model.weight)
    
    optimizer = ScheduleFreeMuon(model.parameters(), lr=0.01)
    
    # Target function: Orthogonal matrix
    # Muon constrains weights to orthogonality, so target MUST be orthogonal
    H = torch.randn(dim, dim, device=device)
    W_true, _ = torch.linalg.qr(H)
    
    print(f"{'Step':<8} {'Loss':<12}")
    print("-" * 20)
    
    model.train()
    for step in range(1000):
        optimizer.zero_grad()
        x = torch.randn(64, dim, device=device)
        y_true = x @ W_true.T
        y_pred = model(x)
        loss = F.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        if step % 100 == 0:
            print(f"{step:<8} {loss.item():<12.6f}")
            
    optimizer.normalize_averaged_weights()
    model.eval()
    x_test = torch.randn(100, dim, device=device)
    y_test = x_test @ W_true.T
    y_pred = model(x_test)
    final_loss = F.mse_loss(y_pred, y_test).item()
    
    print(f"\nFinal Test Loss: {final_loss:.6f}")
    if final_loss < 0.2:
        print("✅ Convergence Test Passed")
    else:
        print("❌ Convergence Test Failed (Loss too high)")

def main():
    parser = argparse.ArgumentParser(description="Schedule-Free Muon Test Suite")
    parser.add_argument('--test', type=str, default='all', choices=['all', 'manifold', 'cost', 'convergence'],
                        help='Which test to run')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to run on')
    
    args = parser.parse_args()
    
    if args.test in ['all', 'manifold']:
        success = test_manifold(args)
        if not success and args.test == 'all':
            print("\n❌ Aborting remaining tests due to failure.")
            return

    if args.test in ['all', 'cost']:
        test_cost(args)
        
    if args.test in ['all', 'convergence']:
        test_convergence(args)

if __name__ == "__main__":
    main()