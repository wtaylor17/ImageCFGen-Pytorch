from image_scms.audio_mnist import *

if __name__ == "__main__":
    spec = torch.zeros((10, *IMAGE_SHAPE))
    latent = torch.randn((10, LATENT_DIM))
    attrs = torch.randn((10, ATTRIBUTE_COUNT))
    enc = Encoder()
    dec = Generator()
    disc = Discriminator()

    codes = enc(spec, [attrs])
    print(f'E(x,c) shape is {codes.shape}')
    rec = dec(codes, [attrs])
    print(f'G(E(x,c),c) shape is {rec.shape}')

    gen = dec(latent, [attrs])
    print(f'G(z,c) shape is {gen.shape}')

    scores = disc(spec, codes, [attrs])
    print(f'D(x,z,c) shape is {scores.shape}')
