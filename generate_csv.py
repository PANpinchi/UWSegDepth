import os
import sys
uw_depth_path = os.path.abspath("./SADDER")
sys.path.insert(0, uw_depth_path)

import csv
from argparse import ArgumentParser

import warnings
warnings.filterwarnings("ignore")


def generate_csv(data_root="./processed_video", output_folder="./outputs"):
    """
    Recursively walk through the 'data/flsea' directory,
    locate paired .tiff or .tif files in 'imgs/' and 'depth/' subfolders,
    then write separate CSVs for train and test data.

    Train/Test split is based on predefined folders:
      - Test: 'data/flsea/canyons/u_canyon', 'data/flsea/red_sea/sub_pier'
      - Train: All other directories

    Output CSV format (one row per matched pair):
        <img_path>,<depth_path>,<features_path>
    """

    out_csv = os.path.join(output_folder, "inference.csv")

    # Temporary storage for file matching
    pairs = {}

    # Recursively search all subdirectories of flsea_root
    valid_exts = ('.tif', '.tiff', '.png', '.jpg')
    image_files = [
        f for f in os.listdir(data_root)
        if f.lower().endswith(valid_exts) and os.path.isfile(os.path.join(data_root, f))
    ]

    for file_name in image_files:
        full_path = os.path.join(data_root, file_name)
        base_name = os.path.splitext(file_name)[0]
        pairs.setdefault(base_name, {})["img"] = full_path

    # Open train and test CSV files for writing
    with open(out_csv, "w", newline="") as out_f:
        out_writer = csv.writer(out_f)

        for base_name, data_dict in sorted(pairs.items()):
            img_path = data_dict["img"]
            img_name = os.path.basename(img_path)

            depth_path = f"{output_folder}/pred_depth/{img_name}"
            depth_path = os.path.splitext(depth_path)[0] + "_depth.tif"

            features_path = f"{output_folder}/matched_features/{img_name}"
            features_path = os.path.splitext(features_path)[0] + "_features.csv"

            segms_path = f"{output_folder}/matched_segms/{img_name}"
            segms_path = os.path.splitext(segms_path)[0] + "_segms.npy"

            # Determine if the path belongs to test or train set
            out_writer.writerow([img_path, depth_path, features_path, segms_path])

    print(f"Inference data saved to {out_csv}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default="./processed_video",
    )
    parser.add_argument("--output_folder",
                        type=str,
                        default="./outputs")
    args = parser.parse_args()
    generate_csv(data_root=args.data_root, output_folder=args.output_folder)
