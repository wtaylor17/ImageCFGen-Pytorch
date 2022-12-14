from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

from image_scms.training_utils import batchify


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--epochs', type=int,
                    default=10)
parser.add_argument('--batch-size', type=int,
                    default=512)


class MNISTClassifier(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(1, 32, (3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, (3, 3)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, (3, 3), (2, 2)),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(4096, 10)
        )


if __name__ == '__main__':
    args = parser.parse_args()
    epochs = args.epochs
    batch_size = args.batch_size

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-train.npy')
    )).float().reshape((-1, 1, 28, 28)).to(device) / 255.0
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)[:, :10]

    criterion = nn.CrossEntropyLoss()

    model = MNISTClassifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    for e in range(epochs):
        batches = list(batchify(x_train, a_train, batch_size=batch_size))
        tq = tqdm(batches)
        for x, y in tq:
            opt.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            opt.step()

            pred = pred.argmax(dim=1)
            y = y.argmax(dim=1)
            acc = torch.eq(pred, y).float().mean()
            tq.set_postfix(dict(loss=loss.item(), acc=acc.item()))

    torch.save({
        "clf": model
    }, "mnist_clf.tar")
