from argparse import ArgumentParser
import os

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from image_scms import mnist as image_mnist
from attribute_scms.mnist import GroundTruthCausalGraph, MNISTCausalGraph
from tqdm import tqdm
from morphomnist.morpho import ImageMorphology, ImageMoments


def extract_observed_attributes(image):
    if type(image) is torch.Tensor:
        image = image.detach().cpu().numpy()
    image = image.reshape((28, 28))
    morph = ImageMorphology(image, scale=16)
    thickness = morph.mean_thickness
    img_min, img_max = image.min(), image.max()
    intensity = np.median(image[image >= img_min + (img_max - img_min) * .5])
    moments = ImageMoments(image)
    slant = -moments.horizontal_shear
    return np.array([thickness, intensity, slant])


parser = ArgumentParser()
parser.add_argument('--data-dir', type=str,
                    help='path to folder with .npy files of data',
                    default='')
parser.add_argument('--image-model',
                    type=str,
                    help='file (.tar) for saved cnn models')
parser.add_argument('--attribute-model',
                    type=str,
                    help='file (.tar) for saved attribute models')
parser.add_argument('--model-name',
                    type=str,
                    default='ImageCFGen')

np.random.seed(42)
torch.manual_seed(42)

if __name__ == '__main__':
    sns.set()
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    a_train = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-train.npy')
    )).float().to(device)
    a_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-a-test.npy')
    )).float().to(device)
    x_test = torch.from_numpy(np.load(
        os.path.join(args.data_dir, 'mnist-x-test.npy')
    )).float().to(device)

    a_train = {
        "digit": a_train[:, :10].float(),
        "thickness": a_train[:, 10:11].float(),
        "intensity": a_train[:, 11:12].float(),
        "slant": a_train[:, 12:13].float()
    }
    a_test = {
        "digit": a_test[:, :10].float(),
        "thickness": a_test[:, 10:11].float(),
        "intensity": a_test[:, 11:12].float(),
        "slant": a_test[:, 12:13].float()
    }

    attr_stats = {
        k: (v.min(dim=0).values, v.max(dim=0).values)
        for k, v in a_train.items()
        if k != "digit"
    }

    ground_truth_scm = GroundTruthCausalGraph()
    trained_graph: MNISTCausalGraph = torch.load(args.attribute_model, map_location=device)['graph']
    trained_graph.get_module('thickness').eval()

    model_dict = torch.load(args.image_model, map_location=device)
    if 'vae' in args.image_model:
        generator = model_dict["vae"].decoder
        encoder = lambda *arg: model_dict["vae"].encoder(*arg)[0]
    else:
        generator = model_dict["G"]
        encoder = model_dict["E"]

    with torch.no_grad():
        x_test = 2 * x_test.reshape((-1, 1, 28, 28)).to(device) / 255.0 - 1
        fig, axs = plt.subplots(1, 3)
        for i, attribute in enumerate(['thickness', 'intensity', 'slant']):
            cf_int = {
                attribute: ground_truth_scm.sample(n=len(x_test))[attribute].reshape((-1, 1))
            }
            cf_real = ground_truth_scm.sample_cf(a_test, cf_int)
            cf_approx = trained_graph.sample_cf(a_test, cf_int)

            a_test_scaled = {
                k: 2 * (v - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
                for k, v in a_test.items()
                if k != "digit"
            }
            a_test_scaled["digit"] = a_test["digit"]
            codes = encoder(x_test, a_test_scaled)

            cf_approx_scaled = {
                k: 2 * (v - attr_stats[k][0]) / (attr_stats[k][1] - attr_stats[k][0]) - 1
                for k, v in cf_approx.items()
                if k != "digit"
            }
            cf_approx_scaled["digit"] = cf_approx["digit"]
            pred_image_out = generator(codes, cf_approx_scaled)

            measured_pred = np.row_stack([
                extract_observed_attributes(255 * (pred_image + 1) / 2)
                for pred_image in tqdm(pred_image_out, total=len(x_test))
            ])

            a_range = torch.range(float(attr_stats[attribute][0]) - 0.2,
                                  float(attr_stats[attribute][1]) + 0.2, 0.01)

            axs[i].plot(a_range, a_range, 'k--')

            axs[i].scatter(cf_real[attribute].cpu().numpy(),
                           measured_pred[:, i], c='blue', alpha=0.7)
            axs[i].set_xlabel('Target value')
            axs[i].set_ylabel('Measured value')
            axs[i].set_title(attribute.capitalize())
        fig.suptitle(args.model_name + ' Morpho-MNIST CF attribute measurement')
        plt.show()
