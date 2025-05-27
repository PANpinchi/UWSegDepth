import sys
import os
import argparse

mmdet_path = os.path.abspath("./BARIS-ERA")
sys.path.insert(0, mmdet_path)

import torch
from mmdet.apis.inference import init_detector, inference_detector, show_result_pyplot
import numpy as np
import cv2
import mmcv
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")

# output shapes
out_height = 240
out_width = 320

# output
vis_folder = "vis_segms"
rel_folder = "matched_segms"

PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100)]


def vis_infer(
        checkpoints="./outputs_swin_base_ours/epoch_11.pth",
        config="./configs/_ours_/mask_rcnn_swin-b-p4-w7_fpn_1x_coco.py",
        data_dir='./processed_video/',
        output_dir='./outputs/',
        num_segms=30
):
    """
    Function to run the DetInferencer for visual inference and save segmentation masks.
    """
    model = init_detector(config, checkpoints, device='cuda:0')

    image_files = [f for f in os.listdir(data_dir) if f.endswith(('.jpg', '.png', '.jpeg', '.tiff'))]

    max_segms = 0
    total_segms = 0
    count = 0

    for img_name in tqdm(image_files):
        img_path = os.path.join(data_dir, img_name)
        try:
            result = inference_detector(model, img_path)

            # visualization
            out_file = os.path.join(output_dir, vis_folder, img_name)
            show_result_pyplot(model,
                               img_path,
                               result,
                               score_thr=0.8,
                               out_file=out_file,
                               palette=PALETTE,
                               is_draw_bbox=False,
                               is_draw_labels=False)

            # Extract segmentation masks
            if isinstance(result, tuple):
                bbox_result, segm_result = result
                if isinstance(segm_result, tuple):
                    segm_result = segm_result[0]  # ms rcnn
            else:
                bbox_result, segm_result = result, None
            bboxes = np.vstack(bbox_result) if bbox_result else []
            labels = [
                np.full(bbox.shape[0], i, dtype=np.int32)
                for i, bbox in enumerate(bbox_result)
            ] if bbox_result else []
            labels = np.concatenate(labels) if labels else []

            # draw segmentation masks
            segms = None
            if segm_result is not None and len(labels) > 0:  # non empty
                segms = mmcv.concat_list(segm_result)
                if isinstance(segms[0], torch.Tensor):
                    segms = torch.stack(segms, dim=0).detach().cpu().numpy().astype(np.uint8)
                else:
                    segms = np.stack(segms, axis=0).astype(np.uint8)

                # Update max segmentation mask count
                max_segms = max(max_segms, segms.shape[0])
                total_segms += segms.shape[0]
                count += 1

                # Resize segmentation masks to output shape
                segms = np.array(
                    [cv2.resize(seg.astype(np.uint8), (out_width, out_height), interpolation=cv2.INTER_NEAREST) for seg
                     in segms])

                if segms.shape[0] > num_segms:
                    segms = segms[:num_segms]
                elif segms.shape[0] < num_segms:
                    padding = np.zeros((num_segms - segms.shape[0], out_height, out_width), dtype=np.uint8)
                    segms = np.vstack((segms, padding))

                # Save segmentation masks only if available
                if segm_result is not None:
                    out_dir = os.path.join(output_dir, rel_folder)
                    os.makedirs(out_dir, exist_ok=True)
                    out_file = os.path.join(out_dir, img_name.replace(os.path.splitext(img_name)[-1], '_segms.npy'))
                    np.save(out_file, segms)
                else:
                    print('segm_result is None')
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

    return max_segms, total_segms, count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./processed_video/")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--model", type=str, default="Swin", choices=["Swin", "ConvNeXt"])
    args = parser.parse_args()

    swin_config = {
        "checkpoints": "./BARIS-ERA/pretrained/baris-era_swin_base.pth",
        "config": "./BARIS-ERA/configs/_ours_/ablation/mask_rcnn_swin-b-p4-w7_fpn_1x_coco_pr2.py",
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
    }
    
    convnext_config = {
        "checkpoints": "./BARIS-ERA/pretrained/baris-era_convnext_base.pth",
        "config": "./BARIS-ERA/configs/_ours_/mask_rcnn_convnext-b_p4_w7_fpn_1x_coco.py",
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
    }

    config = swin_config if args.model == "Swin" else convnext_config

    curr_max_segms = vis_infer(
        checkpoints=config["checkpoints"],
        config=config["config"],
        data_dir=config["data_dir"],
        output_dir=config["output_dir"],
    )

    print('Done!')
