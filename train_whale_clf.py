from classifiers import whalecalls
import torch

if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"
    n_epochs = 100
    l_rate = 1e-4
    batch_size = 32

    clf = whalecalls.train("WhaleData/Nocall", "WhaleData/Gunshot", "WhaleData/Upcall",
                           n_epochs=n_epochs, device=device, batch_size=batch_size,
                           l_rate=l_rate)

    torch.save({
        "clf": clf
    }, "whalecall_clf.tar")
