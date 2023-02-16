from PIL import Image, ImageDraw
import pickle
import argparse
import numpy as np
from tabulate import tabulate

parser = argparse.ArgumentParser()
parser.add_argument("--matrix", type=str)
parser.add_argument("--metric", type=str, default="ssim")

if __name__ == '__main__':
    args = parser.parse_args()
    matrix_path = args.matrix
    metric = args.metric

    with open(matrix_path, "rb") as fp:
        matrix_dicts = pickle.load(fp)


    def get_original_class(i):
        return (set(range(10)) - set(matrix_dicts[i][metric].keys())).pop()


    original_classes = np.array(
        list(map(get_original_class, range(len(matrix_dicts))))
    )


    def aggregate_measure(original_class, target_class):
        inds = np.array(list(range(len(matrix_dicts))))[original_classes == original_class]
        data = np.array([
            matrix_dicts[i][metric][target_class][1]
            for i in inds
            if matrix_dicts[i][metric][target_class] is not None
        ])
        return data.mean()


    measures = {
        i: np.array([aggregate_measure(i, j) if i != j else 0
                     for j in range(10)])
        for i in range(10)
    }

    print(metric, "matrix:")
    print(tabulate(measures, headers=list(range(10)), showindex=list(range(10))))

    print("summary (top 3 for each class):")
    for i in range(10):
        best = measures[i].argsort()
        print(f"{i}: {','.join(map(str, best[best != i][:3]))}")
