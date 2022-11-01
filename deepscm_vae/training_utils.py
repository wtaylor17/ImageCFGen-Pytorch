import torch


def batchify(*tensors, batch_size=256, device='cpu'):
    i = 0
    N = min(map(len, tensors))
    while i < N:
        yield tuple(x[i:i+batch_size] for x in tensors)
        i += batch_size
    if i < N:
        yield tuple(x[i:N].to(device) for x in tensors)


def init_weights(layer):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=0.0001)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)
