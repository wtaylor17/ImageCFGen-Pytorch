from image_scms.audio_mnist import *

if __name__ == "__main__":
    spec = torch.zeros((10, 1, 130, 130))
    attrs = torch.zeros((10, 46))
    enc = Encoder()
    dec = Generator()
    disc = Discriminator()

    latent = enc(spec, attrs)
    print(f'E(x, a) shape is {latent.shape}')
    rec = dec(latent, attrs)
    print(f'G(E(x, a), a) shape is {rec.shape}')

    scores = disc(rec, latent, attrs)
    print(f'D(G(E(x, a), a), E(x, a), a) shape is {scores.shape}')
