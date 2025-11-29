import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def test_cost(args):
    """Benchmark computational cost vs AdamW."""
    print("\n" + "="*60)
    print(" COMPUTATIONAL COST BENCHMARK")
    print("="*60)
    
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    
    # Setup: 6-layer Transformer Encoder
    d_model = 512
    model = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(d_model=d_model, nhead=8, batch_first=True),
        num_layers=6
    ).to(device)
    
    batch_size = 32
    seq_len = 128
    data = torch.randn(batch_size, seq_len, d_model, device=device)
    
    optimizers = {
        'AdamW': torch.optim.AdamW(model.parameters(), lr=1e-3),
        'SF-Muon': ScheduleFreeMuon(model.parameters(), lr=0.02)
    }
    
    results = {}
    
    for name, opt in optimizers.items():
        print(f"\nBenchmarking {name}...")
        
        # Warmup
        for _ in range(10):
            opt.zero_grad()
            out = model(data)
            loss = out.mean()
            loss.backward()
            opt.step()
        
        # Timing
        start_time = time.time()
        steps = 10
        for _ in range(steps):
            opt.zero_grad()
            out = model(data)
            loss = out.mean()
            loss.backward()
            opt.step()
            if device == 'cuda':
                torch.cuda.synchronize()
        
        end_time = time.time()
        duration = end_time - start_time
        steps_per_sec = steps / duration
        results[name] = steps_per_sec
        print(f"  Speed: {steps_per_sec:.2f} steps/sec")
    
    print("\nSummary:")
    base_speed = results['AdamW']
    muon_speed = results['SF-Muon']
    slowdown = base_speed / muon_speed
    print(f"  AdamW:   {base_speed:.2f} steps/sec")
    print(f"  SF-Muon: {muon_speed:.2f} steps/sec")
    print(f"  Slowdown: {slowdown:.2f}x")

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