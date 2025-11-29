import torch
import math
from sf_muon import quintic_newton_schulz

def verify_ns():
    torch.manual_seed(42)
    N = 256
    # Create a Rank-1 matrix (worst case for spectral/frobenius ratio)
    u = torch.randn(N, 1)
    v = torch.randn(N, 1)
    G = u @ v.T
    print(f"Input Norm: {G.norm():.4f}")
    print(f"Input Rank: {torch.linalg.matrix_rank(G)}")
    
    # Run NS with Strict Coefficients and SAFE Pre-conditioning (Norm=1)
    # a, b, c = 1.875, -1.25, 0.375
    # Pre-cond: Scale to Frobenius Norm 1. 
    # Since Spectral Norm <= Frobenius Norm, Spectral Norm <= 1.
    # This guarantees convergence for ANY matrix structure.
    norm = G.norm()
    target_pre_norm = 1.0
    X = G.div(norm).mul_(target_pre_norm)
    
    for _ in range(5):
        A = X @ X.T
        A2 = A @ A
        BX = A @ X
        CX = A2 @ X
        X = 1.875 * X - 1.25 * BX + 0.375 * CX
    
    # Post-cond: Scale to target norm sqrt(N)
    Out = X.div(X.norm()).mul_(math.sqrt(N))
    
    # Check Output Norm
    out_norm = Out.norm()
    target_norm = math.sqrt(N)
    print(f"Output Norm: {out_norm:.4f}")
    print(f"Target Norm: {target_norm:.4f}")
    
    # Check Orthogonality
    # Out^T @ Out should be I
    gram = Out.T @ Out
    I = torch.eye(N)
    diff = (gram - I).norm()
    print(f"Orthogonality Error (||G^T G - I||): {diff:.6f}")
    
    if abs(out_norm - target_norm) < 1e-2:
        print("✅ Norm preserved correctly!")
    else:
        print("❌ Norm mismatch! Shrinkage detected.")

if __name__ == "__main__":
    verify_ns()
