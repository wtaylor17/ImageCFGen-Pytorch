from image_scms.esrf_acoustic import train
import torch
import os
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--data-dir', type=str)
parser.add_argument('--labels', type=str)
parser.add_argument('--epochs', type=int, default=500)
parser.add_argument('--output-path', type=str, default='./esrf-bigan')

if __name__ == '__main__':
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.output_path, exist_ok=True)

    E, G, D, *_ = train(args.data_dir, args.labels,
                        n_epochs=args.epochs,
                        device=device,
                        image_output_path=args.output_path,
                        save_images_every=1)

    torch.save({
        'E': E,
        'G': G,
        'D': D
    }, './esrf-bigan.tar')
