"""
Geometric Stress Test: Manifold Recovery from Random Initialization

This script validates the core geometric claim of Schedule-Free Muon:
"The optimizer can project randomly initialized weights onto the Stiefel 
manifold through continuous orthogonalization during training."

Test Design:
    1. Initialize weights with Gaussian noise (far from manifold)
    2. Train on synthetic regression task
    3. Measure spectral error: ||W^T W - I||_F
    4. Validate convergence to orthogonality

Success Criteria:
    - Final spectral error < 1e-3: Excellent (machine precision)
    - Final spectral error < 0.1: Acceptable (practical use)
    - Final spectral error > 0.1: Failed (did not converge)

Author: PhD Candidate Research Team
"""

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sf_muon import ScheduleFreeMuon


def get_spectral_error(weight_matrix):
    """
    Compute deviation from orthogonality.
    
    For orthogonal matrix W: W^T W = I
    Error measures distance from this ideal.
    
    Args:
        weight_matrix (torch.Tensor): 2D matrix to evaluate
    
    Returns:
        float: Frobenius norm ||W^T W - I||_F
    """
    device = weight_matrix.device
    gram = weight_matrix.T @ weight_matrix
    identity = torch.eye(weight_matrix.shape[1], device=device)
    return torch.norm(gram - identity, p='fro').item()


def run_stress_test(dim=1024, steps=300, device='cuda'):
    """
    Execute geometric stress test.
    
    Args:
        dim (int): Matrix dimension (use 1024+ to see spectral properties)
        steps (int): Training iterations
        device (str): 'cuda' or 'cpu'
    
    Returns:
        bool: True if test passed, False otherwise
    """
    print("\n" + "="*70)
    print(" GEOMETRIC STRESS TEST: MANIFOLD RECOVERY FROM GAUSSIAN NOISE")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Matrix Dimension: {dim}×{dim}")
    print(f"  Training Steps: {steps}")
    print(f"  Device: {device}")
    print(f"  Initialization: N(0, 0.02) - Far from manifold")
    
    # Check device availability
    if device == 'cuda' and not torch.cuda.is_available():
        print("\n⚠️  CUDA not available, falling back to CPU")
        device = 'cpu'
    
    # =========================================================================
    # SETUP: Target and Model
    # =========================================================================
    
    # 1. Ground truth: Random orthogonal matrix (via QR decomposition)
    print("\n[1/6] Generating ground truth orthogonal matrix...")
    H = torch.randn(dim, dim, device=device)
    W_target, _ = torch.linalg.qr(H)
    
    # Sanity check
    target_err = get_spectral_error(W_target)
    print(f"      Target spectral error: {target_err:.2e} (should be ~0)")
    
    # 2. Model: Single linear layer (unit test)
    print("\n[2/6] Initializing model with Gaussian noise...")
    model = nn.Linear(dim, dim, bias=False).to(device)
    nn.init.normal_(model.weight, mean=0, std=0.02)
    
    initial_err = get_spectral_error(model.weight.detach())
    print(f"      Initial spectral error: {initial_err:.4f}")
    print(f"      Expected range: [1.0, 2.0] for Gaussian matrices")
    
    # Validation
    if initial_err < 0.5:
        print("      ⚠️  Warning: Init too close to manifold, test may be trivial")
    
    # =========================================================================
    # OPTIMIZATION
    # =========================================================================
    
    print("\n[3/6] Initializing Schedule-Free Muon optimizer...")
    optimizer = ScheduleFreeMuon(
        model.parameters(),
        lr=0.02,              # Standard Muon LR
        weight_decay=0.01,
        warmup_steps=50
    )
    
    print("\n[4/6] Starting optimization...")
    print(f"      {'Step':<8} {'Loss':<12} {'Spectral Error':<15}")
    print(f"      {'-'*8} {'-'*12} {'-'*15}")
    
    losses = []
    errors = []
    
    model.train()
    for step in range(steps):
        optimizer.zero_grad()
        
        # Synthetic task: Learn to apply W_target
        # y_true = x @ W_target^T
        x = torch.randn(128, dim, device=device)
        y_true = x @ W_target.T
        y_pred = model(x)
        
        loss = nn.functional.mse_loss(y_pred, y_true)
        loss.backward()
        optimizer.step()
        
        # Logging: Dense early, sparse later
        if step < 50 or step % 25 == 0:
            err = get_spectral_error(model.weight.detach())
            losses.append(loss.item())
            errors.append(err)
            
            if step % 50 == 0 or step < 10:
                print(f"      {step:<8} {loss.item():<12.6f} {err:<15.6f}")
    
    # =========================================================================
    # FINAL PROJECTION & EVALUATION
    # =========================================================================
    
    print("\n[5/6] Applying final inference projection...")
    optimizer.normalize_averaged_weights()
    
    final_err = get_spectral_error(model.weight.detach())
    final_loss = losses[-1] if losses else float('nan')
    
    print(f"      Final spectral error: {final_err:.6f}")
    print(f"      Final loss: {final_loss:.6f}")
    
    # =========================================================================
    # VERDICT
    # =========================================================================
    
    print("\n[6/6] Test Results:")
    print("="*70)
    
    if final_err < 1e-3:
        print("✅ [EXCELLENT] Converged to manifold within machine precision")
        print("   Your projection scheme is geometrically sound.")
        print(f"   Error reduction: {initial_err:.4f} → {final_err:.6f}")
        passed = True
    elif final_err < 0.1:
        print("✅ [GOOD] Converged to manifold vicinity")
        print("   Acceptable for practical use.")
        print(f"   Error reduction: {initial_err:.4f} → {final_err:.4f}")
        passed = True
    else:
        print("❌ [FAILED] Did not recover orthogonality")
        print(f"   Final error {final_err:.4f} exceeds threshold 0.1")
        print("   Debug checklist:")
        print("   - Verify Newton-Schulz implementation")
        print("   - Check learning rate (try increasing to 0.1)")
        print("   - Ensure projection is being called every step")
        passed = False
    
    print("="*70)
    
    # =========================================================================
    # VISUALIZATION
    # =========================================================================
    
    print("\n[PLOTTING] Generating diagnostic plots...")
    try:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 1: Loss trajectory
        ax1.plot(losses, 'b-', linewidth=2, alpha=0.7)
        ax1.set_yscale('log')
        ax1.set_xlabel('Logged Steps', fontsize=12)
        ax1.set_ylabel('MSE Loss', fontsize=12)
        ax1.set_title('Training Loss (Log Scale)', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Spectral error trajectory
        ax2.plot(errors, 'r-', linewidth=2, alpha=0.7, label='Spectral Error')
        ax2.axhline(y=1e-3, color='green', linestyle='--', linewidth=2, 
                   label='Excellence Threshold (1e-3)')
        ax2.axhline(y=0.1, color='orange', linestyle='--', linewidth=2,
                   label='Acceptable Threshold (0.1)')
        ax2.set_yscale('log')
        ax2.set_xlabel('Logged Steps', fontsize=12)
        ax2.set_ylabel('||W^T W - I||_F', fontsize=12)
        ax2.set_title('Spectral Error: Distance to Manifold', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = 'stress_test_results.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"   Plot saved to: {plot_path}")
        
        # Try to display (works in notebooks, ignored in scripts)
        try:
            plt.show()
        except:
            pass
            
    except Exception as e:
        print(f"   ⚠️  Plotting failed: {e}")
    
    return passed


def run_comparison_test():
    """
    Compare orthogonal initialization vs random initialization.
    
    Validates that the optimizer works regardless of starting point.
    """
    print("\n" + "="*70)
    print(" COMPARISON TEST: ORTHOGONAL VS RANDOM INITIALIZATION")
    print("="*70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dim = 512
    
    results = {}
    
    for init_type in ['orthogonal', 'random']:
        print(f"\n--- Testing {init_type.upper()} initialization ---")
        
        # Setup
        model = nn.Linear(dim, dim, bias=False).to(device)
        if init_type == 'orthogonal':
            nn.init.orthogonal_(model.weight)
        else:
            nn.init.normal_(model.weight, mean=0, std=0.02)
        
        initial_err = get_spectral_error(model.weight.detach())
        print(f"Initial error: {initial_err:.4f}")
        
        # Train for 100 steps (quick test)
        W_target = torch.linalg.qr(torch.randn(dim, dim, device=device))[0]
        optimizer = ScheduleFreeMuon(model.parameters(), lr=0.02)
        
        for _ in range(200):
            optimizer.zero_grad()
            x = torch.randn(64, dim, device=device)
            y_true = x @ W_target.T
            y_pred = model(x)
            loss = nn.functional.mse_loss(y_pred, y_true)
            loss.backward()
            optimizer.step()
        
        optimizer.normalize_averaged_weights()
        final_err = get_spectral_error(model.weight.detach())
        
        print(f"Final error: {final_err:.6f}")
        results[init_type] = {
            'initial': initial_err,
            'final': final_err,
            'passed': final_err < 0.1
        }
    
    print("\n" + "="*70)
    print("COMPARISON RESULTS:")
    print("="*70)
    for init_type, res in results.items():
        status = "✅ PASS" if res['passed'] else "❌ FAIL"
        print(f"{init_type.upper():12} | {res['initial']:8.4f} → {res['final']:8.6f} | {status}")
    
    return all(res['passed'] for res in results.values())


if __name__ == "__main__":
    # Run main stress test
    passed_main = run_stress_test(dim=1024, steps=300)
    
    # Run comparison test
    print("\n\n")
    passed_comparison = run_comparison_test()
    
    # Final summary
    print("\n" + "="*70)
    print(" FINAL TEST SUMMARY")
    print("="*70)
    print(f"  Main Stress Test:       {'✅ PASSED' if passed_main else '❌ FAILED'}")
    print(f"  Comparison Test:        {'✅ PASSED' if passed_comparison else '❌ FAILED'}")
    print("="*70)
    
    if passed_main and passed_comparison:
        print("\n🎉 All tests passed! Optimizer is geometrically sound.")
    else:
        print("\n⚠️  Some tests failed. Review implementation.")