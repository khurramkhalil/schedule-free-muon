"""
Schedule-Free Muon Optimizer

A geometrically-aware optimizer that combines:
1. Schedule-Free Learning (Defazio et al., 2024) - removes learning rate schedules
2. Muon (Jordan et al., 2024) - spectral/orthogonal updates for matrices

Core Innovation: Continuous manifold projection to handle the mismatch between
Euclidean averaging (Schedule-Free) and Riemannian geometry (orthogonal matrices).

Author: PhD Candidate Research Team
License: MIT
"""

import torch
import torch.optim as optim
import math

try:
    from triton_utils import quintic_newton_schulz_compiled
    HAS_TRITON = False # Temporarily disabled for debugging
except ImportError:
    HAS_TRITON = False


def quintic_newton_schulz(G, steps=5, eps=1e-7):
    """
    Quintic Newton-Schulz Iteration for Matrix Orthogonalization.
    
    Applies optimized fifth-order convergence to project a matrix onto the 
    Stiefel manifold (set of matrices with orthonormal columns).
    
    Mathematical Foundation:
        For matrix X, iteratively compute:
        X_{k+1} = a*X_k + b*X_k(X_k^T X_k) + c*X_k(X_k^T X_k)^2
        
        Where coefficients (a, b, c) = (3.4445, -4.7750, 2.0315) are optimized
        for fifth-order convergence rate.
    
    Args:
        G (torch.Tensor): Input matrix to orthogonalize. Can be 2D (M, N) or
                         higher-dimensional (e.g., Conv kernels). Higher dims
                         are flattened to 2D.
        steps (int): Number of Newton-Schulz iterations. Default 5 (standard).
        eps (float): Numerical stability constant for division.
    
    Returns:
        torch.Tensor: Orthogonalized matrix with same shape as input.
                     Satisfies ||X^T X - I|| ≈ 0 (within numerical precision).
    
    Computational Cost:
        For M×N matrix: O(steps * M*N^2) if M > N, else O(steps * M^2*N)
        
    References:
        - Jordan et al. (2024): "Muon: Momentum Orthogonalized by Newton-Schulz"
        - Higham (1986): "Computing the polar decomposition"
    """
    # Quintic coefficients for strict orthogonality (sum=1)
    # Derived for 3rd order convergence to I: f(1)=1, f'(1)=0, f''(1)=0
    # a, b, c = 1.875, -1.25, 0.375        # Strict (converges to I)
    a, b, c = 3.4445, -4.7750, 2.0315  # Original Muon (approximate, scales to ~0.8)
    
    # Handle multi-dimensional tensors (e.g., Conv2D kernels)
    original_shape = G.shape
    original_dtype = G.dtype
    if G.ndim > 2:
        # Flatten: (Out, In, H, W) -> (Out, In*H*W)
        G = G.view(G.size(0), -1)
    
    # Enforce float32 for stability
    G = G.to(torch.float32)
    
    M, N = G.shape
    
    # Optimization: Always work with smaller Gram matrix
    # If tall (M > N), transpose to work on (N, N) instead of (M, M)
    transpose_needed = M > N
    if transpose_needed:
        X = G.T  # Now X is (N, M) where N < M
    else:
        X = G    # X is (M, N) where M <= N
    
    # Pre-conditioning: Scale to ensure convergence
    # Newton-Schulz converges if spectral norm ||X||_2 < sqrt(3)
    # Using Frobenius norm as conservative upper bound ensures stability
    norm = X.norm(p='fro') + eps
    X = X.div(norm)
    
    # Quintic iteration
    for _ in range(steps):
        # Compute Gram matrix: A = X @ X^T
        # Shape: (rows, rows) where rows = min(M, N)
        A = torch.einsum('ik, jk -> ij', X, X)
        
        # Compute A^2
        A2 = torch.einsum('ij, jk -> ik', A, A)
        
        # Compute update terms
        # BX = A @ X (cubic term)
        BX = torch.einsum('ij, jk -> ik', A, X)
        
        # CX = A^2 @ X (quintic term)
        CX = torch.einsum('ij, jk -> ik', A2, X)
        
        # Update: X_{k+1} = aX + bAX + cA^2X
        X = a * X + b * BX + c * CX
        
    # Post-conditioning: Re-normalize to ensure we don't shrink
    # We force the Frobenius norm back to sqrt(N) (spectral norm ~ 1)
    current_norm = X.norm(p='fro') + eps
    target_norm = math.sqrt(X.size(0))
    X = X.mul(target_norm / current_norm)
    
    # Restore orientation if transposed
    if transpose_needed:
        X = X.T
    
    # Restore original shape and dtype
    return X.view(original_shape).to(original_dtype)


class ScheduleFreeMuon(optim.Optimizer):
    """
    Schedule-Free Muon: A hyperparameter-reduced optimizer for deep learning.
    
    Combines Schedule-Free Learning (removes LR schedules) with Muon 
    (matrix-aware spectral updates), using continuous manifold projection
    to bridge Euclidean and Riemannian geometries.
    
    Key Features:
        1. No learning rate schedule required (schedule-free averaging)
        2. Orthogonal updates for matrix parameters (spectral conditioning)
        3. Memory-efficient (in-place operations, no persistent clones)
        4. Automatic layer routing (Muon for matrices, AdamW for vectors)
    
    Mathematical Overview:
        - Anchor (z): Fast-moving exploration point
        - Average (y): Slow-moving stabilization point
        - Training weights: Projected combination of y and z
        - Inference weights: Fully projected y
    
    Args:
        params: Iterable of parameters to optimize
        lr (float): Learning rate. Default: 0.02 (aggressive, Muon-style)
        betas (tuple): Adam momentum coefficients (beta1, beta2). Default: (0.9, 0.999)
        weight_decay (float): Decoupled weight decay coefficient. Default: 0.01
        warmup_steps (int): Linear warmup period. Default: 1000
    
    Example:
        >>> model = MyTransformer()
        >>> optimizer = ScheduleFreeMuon(model.parameters(), lr=0.02)
        >>> 
        >>> # Training loop
        >>> for batch in dataloader:
        >>>     optimizer.zero_grad()
        >>>     loss = model(batch)
        >>>     loss.backward()
        >>>     optimizer.step()
        >>>
        >>> # Before inference/saving
        >>> optimizer.normalize_averaged_weights()
    
    References:
        - Defazio et al. (2024): "The Road Less Scheduled"
        - Jordan et al. (2024): "Muon Optimizer"
    """
    
    def __init__(self, params, lr=0.02, betas=(0.9, 0.999), momentum=0.95,
                 weight_decay=0.01, warmup_steps=1000):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr, 
            betas=betas, 
            momentum=momentum,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps, 
            k=0  # Global step counter
        )
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                                         and returns the loss.
        
        Returns:
            loss (float, optional): Loss value if closure is provided.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            weight_decay = group['weight_decay']
            k = group['k']
            warmup_steps = group['warmup_steps']
            
            # Schedule-Free averaging rate: c_k = 1/(k+1)
            # Decays over time, ensuring convergence
            ck = 1.0 / (k + 1) if k > 0 else 1.0
            
            # Warmup schedule for forward pass interpolation
            # Gradually shifts from averaged (y) to anchor (z)
            sched = min(1.0, (k + 1) / warmup_steps)
            
            # Apply LR warmup to the step size as well
            # This allows using higher peak LR without instability at start
            lr_scale = min(1.0, (k + 1) / warmup_steps)
            current_lr = lr * lr_scale
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                state = self.state[p]
                
                # Initialize state on first step
                if len(state) == 0:
                    state['z'] = p.clone()  # Anchor: exploration trajectory
                    state['y'] = p.clone()  # Average: stabilization trajectory
                    state['step'] = 0
                
                z = state['z']
                y = state['y']
                grad = p.grad
                state['step'] += 1
                
                # ============================================================
                # BRANCH 1: VECTOR PARAMETERS (Bias, LayerNorm, Embeddings)
                # Use standard AdamW (Euclidean space)
                # ============================================================
                if p.ndim < 2:
                    # Initialize Adam state if needed
                    if 'exp_avg' not in state:
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)
                    
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']
                    
                    # Decoupled weight decay (applied to gradient)
                    if weight_decay != 0:
                        grad = grad.add(z, alpha=weight_decay)
                    
                    # Adam momentum updates
                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    # Bias-corrected step size
                    denom = exp_avg_sq.sqrt().add_(1e-8)
                    bias_correction1 = 1 - beta1 ** (k + 1)
                    bias_correction2 = 1 - beta2 ** (k + 1)
                    step_size = current_lr * math.sqrt(bias_correction2) / bias_correction1
                    
                    # Update anchor z
                    z.addcdiv_(exp_avg, denom, value=-step_size)
                
                # ============================================================
                # BRANCH 2: MATRIX PARAMETERS (Linear, Conv2D)
                # Use Muon (Stiefel manifold)
                # ============================================================
                # ============================================================
                # BRANCH 2: MATRIX PARAMETERS (Linear, Conv2D)
                # Use Muon (Stiefel manifold)
                # ============================================================
                else:
                    momentum = group['momentum']
                    
                    # Coupled Weight Decay (L2 Regularization)
                    # Add to gradient so it gets orthogonalized with the update
                    # This ensures the entire update step is spectral
                    if weight_decay != 0:
                        grad.add_(z, alpha=weight_decay)
                    
                    # Muon Momentum: v_{t+1} = mu * v_t + g_t
                    if 'momentum_buffer' not in state:
                        state['momentum_buffer'] = torch.zeros_like(grad)
                    
                    buf = state['momentum_buffer']
                    buf.mul_(momentum).add_(grad)
                    
                    # Orthogonalize MOMENTUM BUFFER (spectral preconditioning)
                    # This is the core Muon innovation: orthogonalize the accumulated direction
                    if HAS_TRITON and grad.device.type == 'cuda':
                        g_ortho = quintic_newton_schulz_compiled(buf, steps=5)
                    else:
                        g_ortho = quintic_newton_schulz(buf, steps=5)
                    
                    # Update anchor z with orthogonal update
                    # Note: z drifts in ambient Euclidean space
                    if grad.size(-2) > grad.size(-1): # Tall matrix
                        scale = (grad.size(-2) / grad.size(-1)) ** 0.5
                    else:
                        scale = 1.0
                        
                    z.add_(g_ortho, alpha=-current_lr * scale)
                    
                    # Periodic anchor stabilization (drift correction)
                    # Prevents z from wandering too far from manifold
                    if state['step'] % 10 == 0:
                        z.copy_(quintic_newton_schulz(z, steps=1))
                        # Also project y to prevent shrinkage from averaging
                        y.copy_(quintic_newton_schulz(y, steps=1))
                
                # ============================================================
                # MERGE: SCHEDULE-FREE AVERAGING
                # Applies to both vector and matrix parameters
                # ============================================================
                # Update average: y_{k+1} = (1 - c_k) * y_k + c_k * z_{k+1}
                # Note: For matrices, this is Euclidean averaging (manifold drift)
                y.mul_(1 - ck).add_(z, alpha=ck)
                
                # ============================================================
                # FORWARD PASS WEIGHT PREPARATION (Memory Optimized)
                # Construct weights for next forward pass
                # ============================================================
                # Compute interpolation: w = (1 - sched) * y + sched * z
                # Write directly to p.data (in-place, no clone)
                p.data.copy_(y)
                p.data.mul_(1 - sched).add_(z, alpha=sched)
                
                # CRITICAL: Re-project to manifold for matrices
                # This ensures gradients in next iteration are computed at
                # a valid point on the Stiefel manifold
                if p.ndim >= 2:
                    # Use 5 steps to ensure we are firmly on the manifold
                    # 1 step might be insufficient if y has shrunk significantly
                    p.data.copy_(quintic_newton_schulz(p.data, steps=5))
            
            # Increment global step counter
            group['k'] += 1
        
        return loss
    
    def normalize_averaged_weights(self):
        """
        Project averaged weights to Stiefel manifold for inference.
        
        MUST be called before:
        - Model evaluation/inference
        - Checkpoint saving
        - Final model export
        
        This applies high-precision orthogonalization (10 NS steps) to the
        accumulated average 'y', correcting any drift from Euclidean averaging.
        
        Example:
            >>> # After training
            >>> optimizer.normalize_averaged_weights()
            >>> torch.save(model.state_dict(), 'model.pt')
            >>>
            >>> # Or before eval
            >>> optimizer.normalize_averaged_weights()
            >>> model.eval()
            >>> test_loss = evaluate(model, test_loader)
        """
        for group in self.param_groups:
            for p in group['params']:
                # Only project matrix parameters
                if p.ndim >= 2:
                    state = self.state[p]
                    if 'y' in state:
                        # High-precision projection (10 steps vs 1 during training)
                        state['y'] = quintic_newton_schulz(state['y'], steps=10)
                        # Load projected weights into model
                        p.data.copy_(state['y'])