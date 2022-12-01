# from image_scms.audio_mnist import *
from gans.audio_mnist import *

if __name__ == "__main__":
    spec = torch.zeros((10, *IMAGE_SHAPE))
    latent = torch.randn((10, LATENT_DIM))
    dec = Generator()
    disc = Discriminator()

    rec = dec(latent)
    print(f'G(z) shape is {rec.shape}')

    scores = disc(spec)
    print(f'D(x) shape is {scores.shape}')
    scores = disc(dec(latent))
    print(f'D(G(z)) shape is {scores.shape}')
