import torch
import torch.nn as nn
from pytorch_msssim import ssim


def batchify(*tensors, batch_size=128, device='cpu'):
    i = 0
    N = min(map(len, tensors))
    while i < N:
        yield tuple(x[i:i+batch_size] for x in tensors)
        i += batch_size
    if i < N:
        yield tuple(x[i:N].to(device) for x in tensors)


def binarized_attribute_channel(image: torch.Tensor, attributes: torch.Tensor, device='cpu'):
    n, c, w, h = image.shape
    out = torch.zeros((n, attributes.shape[1], w, h)).float().to(device)
    labels = attributes.argmax(dim=1)
    out[range(n), labels, :, :] = 1.0
    return out


def attributes_image(image, attributes, device='cpu'):
    # image is (n, c, w, h)
    # attributes is (n, k) where k < h
    n, c, w, h = image.shape
    _, k = attributes.shape
    attr_image = torch.zeros((n, 1, w, h)).float().to(device)

    attr_image[:, :, :, h//2-k//2-k % 2:h//2+k//2] = attributes.reshape((n, 1, 1, k))
    return torch.concat([image.to(device), attr_image], dim=1)


def log_loss(score_0, score_1, eps=1e-6):
    loss = torch.log(score_1 + eps) + torch.log(1 - score_0 + eps)
    return -torch.mean(loss)


class AdversariallyLearnedInference(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 discriminator: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.discriminator = discriminator

    def __call__(self, x, z, a=None, add_noise=False, noise_scale=0.1):
        encoder_args = (x,)
        decoder_args = (z,)
        if a is not None:
            encoder_args = encoder_args + (a,)
            decoder_args = decoder_args + (a,)
        ex = self.encoder(*encoder_args)
        gz = self.decoder(*decoder_args)
        dg_args = (gz, z)
        de_args = (x, ex)
        if add_noise:
            device = next(self.encoder.parameters()).device
            de_args = (x + torch.normal(0, noise_scale, x.shape).to(device), ex)
        if a is not None:
            dg_args = dg_args + (a,)
            de_args = de_args + (a,)

        return self.discriminator(*dg_args), self.discriminator(*de_args)

    def discriminator_loss(self, x, z, a=None, eps=1e-6, **kwargs):
        dg, de = self(x, z, a=a, **kwargs)
        return log_loss(dg, de, eps)

    def generator_loss(self, x, z, a=None, eps=1e-6, **kwargs):
        dg, de = self(x, z, a=a, **kwargs)
        return log_loss(de, dg, eps)

    def rec_loss(self, x, z=None, a=None, metric='ssim'):
        if metric == 'mse':
            def loss(Y, X):
                return torch.square(Y - X).mean()
        elif metric == 'ssim':
            def loss(Y, X):
                return 1 - ssim(Y, X, data_range=1.0, size_average=True)
        else:
            raise ValueError(f'Invalid metric {metric}')

        if z is None:
            encoder_args = (x,)
            if a is not None:
                encoder_args = encoder_args + (a,)
            z = self.encoder(*encoder_args)
        decoder_args = (z,)
        if a is not None:
            decoder_args = decoder_args + (a,)
        rec = self.decoder(*decoder_args)

        return loss(x, rec)


def init_weights(layer, std=0.01):
    name = layer.__class__.__name__
    if name.startswith('Conv'):
        torch.nn.init.normal_(layer.weight, mean=0, std=std)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias, 0)


class LambdaLayer(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, *args, **kwargs):
        return self.fn(*args, **kwargs)
