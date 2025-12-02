import torch
import numpy as np

def to_tensor(x, dtype=torch.float32, device=None):
    if isinstance(x, np.ndarray):
        t = torch.tensor(x, dtype=dtype)
    elif torch.is_tensor(x):
        t = x.type(dtype)
    else:
        t = torch.tensor(x, dtype=dtype)
    if device is not None:
        return t.to(device)
    return t

def to_numpy(t):
    if torch.is_tensor(t):
        return t.detach().cpu().numpy()
    return np.array(t)
