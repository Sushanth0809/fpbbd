import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

def get_scheduler(optimizer, scheduler_type='step', **kwargs):
    """
    Get learning rate scheduler.

    Args:
        optimizer: Optimizer
        scheduler_type: Type of scheduler ('step' or 'cosine')
        **kwargs: Additional arguments for scheduler

    Returns:
        scheduler: LR scheduler
    """
    if scheduler_type == 'step':
        step_size = kwargs.get('step_size', 10)
        gamma = kwargs.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif scheduler_type == 'cosine':
        T_max = kwargs.get('T_max', 50)
        return CosineAnnealingLR(optimizer, T_max=T_max)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")