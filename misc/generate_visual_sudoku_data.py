import os
import pickle
import numpy as np
from PIL import Image
from glob import glob
from tqdm import tqdm
import numpy as np

import torch
import pickle

import torchvision
import torchvision.datasets as datasets
from torchvision import datasets, transforms

from IPython.core.debugger import Pdb


import argparse

parser = argparse.ArgumentParser()

MAX_SAMPLES = 20000
parser.add_argument("--input_files", nargs="+")
parser.add_argument("--data_dir", type=str, default="data")
parser.add_argument("--output_dir", type=str, default="vis_sudoku")
parser.add_argument("--mnist_dir", type=str, default="mnist")
parser.add_argument("--max_samples", type=int, default=MAX_SAMPLES)


args = parser.parse_args()
max_samples = min(MAX_SAMPLES, args.max_samples)
file_names = args.input_files
# file_names = ['sudoku_2x2_1M.pt', 'sudoku_2x3_11214.pt']

data_dir = args.data_dir
output_dir = os.path.join(data_dir, args.output_dir)

os.makedirs(output_dir, exist_ok=True)

mnist_trainset = datasets.MNIST(
    root=os.path.join(data_dir, args.mnist_dir),
    train=True,
    download=True,
    transform=None,
)
x_train = mnist_trainset.data.numpy()
y_train = mnist_trainset.targets.numpy()

mask_0 = np.isin(y_train, 0)
mask_1 = np.isin(y_train, 1)
mask_2 = np.isin(y_train, 2)
mask_3 = np.isin(y_train, 3)
mask_4 = np.isin(y_train, 4)
mask_5 = np.isin(y_train, 5)
mask_6 = np.isin(y_train, 6)
mask_7 = np.isin(y_train, 7)
mask_8 = np.isin(y_train, 8)
mask_9 = np.isin(y_train, 9)

x0 = x_train[mask_0]
x1 = x_train[mask_1]
x2 = x_train[mask_2]
x3 = x_train[mask_3]
x4 = x_train[mask_4]
x5 = x_train[mask_5]
x6 = x_train[mask_6]
x7 = x_train[mask_7]
x8 = x_train[mask_8]
x9 = x_train[mask_9]

x = [x0, x1, x2, x3, x4, x5, x6, x7, x8, x9]

img_size = 28

# class_map = {0: 0, 1: 3, 2: 4, 3: 5, 4: 6, 5: 7, 6: 8, 7: 9, 8: 1, 9: 2}
class_map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9}
for file_name in file_names:
    data = torch.load(os.path.join(data_dir, file_name), map_location="cpu")
    this_max_samples = min(max_samples, len(data["query_tensor"]))
    data["query_tensor"] = data["query_tensor"][:this_max_samples]
    data["target_tensor"] = data["target_tensor"][:this_max_samples]
    x_tensor = data["query_tensor"]
    y = data["target_tensor"]
    grid_size = data["box_shape"][0] * data["box_shape"][1]
    x_image_list = []
    for idx in tqdm(range(this_max_samples)):
        # for idx in tqdm(range(10)):
        query, target = x_tensor[idx], y[idx]
        query_data = np.array(query).reshape(grid_size, grid_size)
        seed_list = []
        sudoku_board_query = []
        for i in range(grid_size):
            # row = np.ndarray([grid_size, img_size, img_size])
            for j in range(grid_size):
                specific_digit_set = x[class_map[int(query_data[i, j])]]
                seed_list.append(np.random.randint(0, 100000))
                np.random.seed(seed_list[-1])
                random_id = np.random.choice(specific_digit_set.shape[0], 1)
                img = specific_digit_set[random_id[0]]
                # row[j] = img
                sudoku_board_query.append(torch.from_numpy(img).byte())
            # sudoku_board_query.append(np.concatenate(row, axis=1))
        # sudoku_board_query = np.concatenate(sudoku_board_query, axis=0).reshape(img_size * grid_size, img_size * grid_size)
        x_image_list.append(torch.stack(sudoku_board_query, dim=0))
        # sudoku_board_query = Image.fromarray(np.uint8(sudoku_board_query), mode='L')
        # sudoku_board_query.save(f"./vis_sudoku/train/query/{idx}.png")
    data["query_image_tensor"] = torch.stack(x_image_list, dim=0)
    torch.save(data, os.path.join(output_dir, file_name))
    sample_image = np.array(
        [
            x[class_map[0]][0].tolist(),
            x[class_map[1]][0].tolist(),
            x[class_map[2]][0].tolist(),
            x[class_map[3]][0].tolist(),
            x[class_map[4]][0].tolist(),
            x[class_map[5]][0].tolist(),
            x[class_map[6]][0].tolist(),
            x[class_map[7]][0].tolist(),
            x[class_map[8]][0].tolist(),
            x[class_map[9]][0].tolist(),
        ]
    )
    #
    np.save(os.path.join(output_dir, "sample_images.npy"), sample_image)
    sample_image = np.concatenate(sample_image, axis=1)
    sample_image = Image.fromarray(np.uint8(sample_image), mode="L")
    sample_image.save(os.path.join(output_dir, "sample.png"))
