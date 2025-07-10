import math
from torch.optim.lr_scheduler import _LRScheduler


class CosineWarmupScheduler(_LRScheduler):
    """
    Cosine annealing learning rate scheduler with warmup.
    
    Args:
        optimizer: Wrapped optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total number of training steps
        last_epoch: The index of last epoch. Default: -1
        verbose: If True, prints a message to stdout for each update. Default: False
    """
    
    def __init__(self, optimizer, warmup_steps, total_steps, last_epoch=-1, verbose=False):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        """Calculate learning rate based on current step."""
        if not self._get_lr_called_within_step:
            import warnings
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        step = self.last_epoch
        lrs = []
        
        for base_lr in self.base_lrs:
            if step < self.warmup_steps:
                # Warmup phase: linear increase
                lr = base_lr * (step + 1) / self.warmup_steps
            else:
                # Cosine annealing phase
                e = step - self.warmup_steps
                es = self.total_steps - self.warmup_steps
                lr = 0.5 * (1 + math.cos(math.pi * e / es)) * base_lr
            
            lrs.append(lr)
        
        return lrs


# Example usage:
if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import matplotlib.pyplot as plt
    
    # Create a simple model and optimizer
    model = nn.Linear(10, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Create scheduler
    warmup_steps = 100
    total_steps = 1000
    scheduler = CosineWarmupScheduler(optimizer, warmup_steps, total_steps)
    
    # Simulate training and collect learning rates
    lrs = []
    for step in range(total_steps):
        # Your training step would go here
        
        # Step the scheduler
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])
    
    # Plot the learning rate schedule
    plt.figure(figsize=(10, 6))
    plt.plot(range(total_steps), lrs, 'b-', linewidth=2)
    plt.axvline(x=warmup_steps, color='r', linestyle='--', alpha=0.7, label='End of warmup')
    plt.xlabel('Step')
    plt.ylabel('Learning Rate')
    plt.title('Cosine Warmup Learning Rate Schedule')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
    
    print(f"Initial LR: {lrs[0]:.6f}")
    print(f"Max LR (after warmup): {lrs[warmup_steps]:.6f}")
    print(f"Final LR: {lrs[-1]:.6f}")