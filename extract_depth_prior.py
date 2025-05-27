"""
# > Script for inferencing UDepth on image/folder/video data
#    - Paper: https://arxiv.org/pdf/2209.12358.pdf
"""
import os
import sys
uw_depth_path = os.path.abspath("./SADDER")
sys.path.insert(0, uw_depth_path)

import cv2
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
# local libs
from udepth_model.udepth import *
from utils.data import *
from utils.utils import *
from CPD.sod_mask import get_sod_mask
import tifffile
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def get_depth(image, model_CPD_path, max_val):
    """Generate depth map"""
    # Prepare SOD mask
    mask = np.array(get_sod_mask(image, model_CPD_path=model_CPD_path))
    # Convert RGB color space into RMI input space if needed
    if args.input_space == "RMI":
        image = RGB_to_RMI(image)
    # Prepare data
    image_tensor = totensor(image).unsqueeze(0)
    input_img = torch.autograd.Variable(image_tensor.to(device=device))
    # Generate depth map
    _, out = net(input_img)
    # Apply guidedfilter to depth map
    result = output_result(out, mask)
    ab_result = output_ab_result(out, max_val)

    return result, ab_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--loc_img", type=str, default="./data/imgs/00002.png")
    parser.add_argument("--loc_folder", type=str, default="./processed_video")
    parser.add_argument("--model_RMI_path", type=str, default="./SADDER/saved_udepth_model/model_RMI.pth")
    parser.add_argument("--model_RGB_path", type=str, default="./SADDER/saved_udepth_model/model_RGB.pth")
    parser.add_argument("--model_CPD_path", type=str, default="./SADDER/CPD/CPD-R.pth")
    parser.add_argument("--input_space", type=str, default="RMI", choices=["RMI", "RGB"])
    parser.add_argument("--data_type", type=str, default="folder", choices=["image", "folder"])
    parser.add_argument("--output_folder", type=str, default="./outputs/")
    args = parser.parse_args()

    # Define input space
    image_space = args.input_space

    # Create output folder if not exist
    rel_folder = "pred_depth"
    out_dir = os.path.join(args.output_folder, rel_folder)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Use cuda
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load specific model
    model_path = args.model_RMI_path if args.input_space == "RMI" else args.model_RGB_path
    net = UDepth.build(n_bins=80, min_val=0.001, max_val=10, norm="linear")
    net.load_state_dict(torch.load(model_path))
    print("Model loaded: UDepth")

    net = net.to(device=device)
    net.eval()

    # Load data for image data type
    if args.data_type == "image":
        img_fn = args.loc_img.split("/")[-1]
        # Load data
        image = Image.open(args.loc_img)
        # Generate depth map
        result, ab_result = get_depth(image, model_CPD_path=args.model_CPD_path)
        # Save result
        plt.imsave(os.path.join(out_dir, img_fn), result, cmap='inferno')

    # Load data for folder data type
    if args.data_type == "folder":
        # Inferencing loop
        valid_exts = ('.tif', '.tiff', '.png', '.jpg')
        image_files = sorted([
            f for f in os.listdir(args.loc_folder)
            if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(args.loc_folder, f))
        ])
        for img_fn in tqdm(image_files):
            # Load data
            img_path = os.path.join(args.loc_folder, img_fn)
            image = Image.open(img_path)
            # Generate depth map
            result, ab_result = get_depth(image, model_CPD_path=args.model_CPD_path, max_val=10)
            # Save result
            # plt.imsave(os.path.join(out_dir, img_fn), result, cmap='inferno')
            ab_result[result < 50] = 0.
            result_to_save = ab_result.astype(np.float32)
            output_path = os.path.join(out_dir, os.path.splitext(img_fn)[0] + "_depth.tif")
            tifffile.imwrite(output_path, result_to_save)
        print("Total images: {0}\n".format(len(image_files)))

