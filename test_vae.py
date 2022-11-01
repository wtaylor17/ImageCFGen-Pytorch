from deepscm_vae.mnist import MorphoMNISTVAE
from deepscm_vae.training_utils import init_weights
import torch


if __name__ == '__main__':
    vae = MorphoMNISTVAE()
    vae.encoder.apply(init_weights)
    vae.decoder.apply(init_weights)
    vae.encoder.eval()
    vae.decoder.eval()
    x = torch.randn((1, 1, 28, 28))
    c = torch.randn((1, 13))

    codes = vae.encoder.sample(x, c)
    print(f'Codes shape: {codes.shape}')

    rec = vae.decoder(codes, c)
    print(f'Reconstruction shape: {rec.shape}')

    elbo = vae.elbo(x, c)
    print(f'ELBO shape: {elbo.shape} item: {elbo.item()}')
