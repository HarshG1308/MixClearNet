from functools import lru_cache
from typing import Any, Callable
import time
import random
import torch
from data_loader import load_dataset

# LRU cache for dataset loading
@lru_cache(maxsize=32)
def cached_dataset_loader(dataset_name: str) -> Any:
    """Load and cache the dataset."""
    return load_dataset(dataset_name)

# Input validation decorator
def validate_inputs(validation_fn: Callable):
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            if not validation_fn(*args, **kwargs):
                raise ValueError("Input validation failed.")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# Reproducibility checklist
def set_random_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# Performance benchmarking
def benchmark_model(model, dataloader, device):
    """Benchmark the model's performance on the given dataloader."""
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            _ = model(inputs)
    end_time = time.time()
    print(f"Benchmarking completed in {end_time - start_time:.2f} seconds.")