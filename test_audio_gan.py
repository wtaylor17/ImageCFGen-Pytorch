from image_scms.audio_mnist import *


def one_hot(x, out_dim=None):
    out_dim = out_dim or x.max()
    hot = torch.zeros((x.size(0), out_dim))
    hot[range(x.size(0)), x] = 1
    return hot.int()


if __name__ == "__main__":
    spec = torch.zeros((10, *IMAGE_SHAPE))
    latent = torch.randn((10, LATENT_DIM))
    attrs = {
        k: one_hot(torch.randint(0, v, size=(10,)), v)
        for k, v in ATTRIBUTE_DIMS.items()
    }
    enc = Encoder()
    dec = Generator()
    disc = Discriminator()

    codes = enc(spec, attrs)
    print(f'E(x,c) shape is {codes.shape}')
    rec = dec(codes, attrs)
    print(f'G(E(x,c),c) shape is {rec.shape}')

    gen = dec(latent, attrs)
    print(f'G(z,c) shape is {gen.shape}')

    scores = disc(spec, codes, attrs)
    print(f'D(x,z,c) shape is {scores.shape}')
