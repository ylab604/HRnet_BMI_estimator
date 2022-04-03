import os
import matplotlib
import argparse
from PIL import Image
from lib.config import config, update_config
import lib.models as models
import torch
import matplotlib.pyplot as plt

import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
# from tiny_unet import UNet
import numpy as np

parser = argparse.ArgumentParser(
    description="PyTorch depth prediction evaluation script"
)
parser.add_argument(
    "model_folder",
    type=str,
    metavar="F",
    help="In which folder have you saved the model",
)
parser.add_argument(
    "--path", type=str, default="data", metavar="D", help="image file path"
)
parser.add_argument(
    "--cfg", help="experiment configuration filename", required=True, type=str
)
parser.add_argument("--model-file", help="model parameters", required=True, type=str)


args = parser.parse_args()

from data import output_height, output_width

state_dict = torch.load(args.model_folder)

model = models.Hrnet2DNnet(config)

model.load_state_dict(state_dict)
model.eval()

img = Image.open(args.path)
img = img.resize((512, 512))
img_np = np.asarray(img)
img_t = torch.from_numpy(img_np)
img_t = img_t.view(1, 3, 512, 512)
img_t = img_t.float()
output = model(img_t)
output = output.detach().numpy()
print(output.shape)
plt.imsave("output.png", np.transpose(output[0][0], (0, 1)))

