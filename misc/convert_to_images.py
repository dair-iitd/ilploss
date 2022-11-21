import torch
from PIL import Image
import numpy as np
import os

data = torch.load("data/vis_sudoku/sudoku_2x2_1M.pt")
output_dir = "data/vis_sudoku/2x2_images"
os.makedirs(output_dir, exist_ok=True)
box = data["box_shape"]
bs = box[0] * box[1]

# for index in range(data['query_image_tensor'].shape[0]):
for index in range(1):
    img = data["query_image_tensor"][index]
    # img.shape == num_cells x 28 x 28
    img_size = img.shape[-1]
    img = img.reshape(bs, bs, img_size, img_size)
    # for row in range(img.shape[0]):
    img = torch.concat(
        [
            torch.concat([img[row][col] for col in range(bs)], dim=1)
            for row in range(bs)
        ],
        dim=0,
    )
    img = Image.fromarray(img.numpy(), mode="L")
    img.save(os.path.join(output_dir, "{}.png".format(index)))
