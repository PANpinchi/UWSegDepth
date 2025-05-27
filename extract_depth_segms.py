import os
import sys
import argparse

uw_depth_path = os.path.abspath("./SADDER")
mmdet_path = os.path.abspath("./BARIS-ERA")
sys.path.insert(0, uw_depth_path)
sys.path.insert(0, mmdet_path)

from os.path import join, basename, splitext
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import mmcv
from mmdet.apis.inference import init_detector, inference_detector

from depth_estimation.model.model import UDFNet
from data.example_dataset.dataset import get_example_dataset_inference

from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
from mmdet.core.mask.structures import bitmap_to_polygon

from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

import matplotlib.pyplot as plt
from mmdet.core.visualization.palette import get_palette, palette_val
from mmdet.core.visualization.image import draw_masks

############################################################
###################### CONFIG ##############################
############################################################

BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEPTH_MODEL_PATH = "./SADDER/saved_models/model_e24_udfnet_lr1e-05_bs6_lrd0.9_with_SADDER.pth"

SEG_SWIN_MODEL_CONFIG = "./BARIS-ERA/configs/_ours_/ablation/mask_rcnn_swin-b-p4-w7_fpn_1x_coco_pr2.py"
SEG_SWIN_MODEL_CKPT = "./BARIS-ERA/outputs_swin_base_ours_pr2_old/epoch_11.pth"

SEG_CONVNEXT_MODEL_CONFIG = "./BARIS-ERA/configs/_ours_/mask_rcnn_convnext-b_p4_w7_fpn_1x_coco.py"
SEG_CONVNEXT_MODEL_CKPT = "./BARIS-ERA/outputs_convnext_base_ours/epoch_12.pth"

CLASSES = ('fish', 'reefs', 'aquatic plants', 'wrecks/ruins', 'human divers', 'robots', 'sea-floor')
PALETTE = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100), (0, 80, 100)]

SCORE_THR = 0.8
DRAW_BBOX = True
DRAW_LABELS = True
DRAW_SCORE = False  # This will only work if DRAW_LABELS = True and DRAW_BBOX = True.
DRAW_DEPTHS = True  # This will only work if DRAW_LABELS = True.
SAVE = True
EPS = 1e-2

############################################################


def show_depth_segms_results(img, result, prediction_map, score_thr=0.3,
                             bbox_color=(72, 101, 241), text_color=(72, 101, 241), mask_color=None,
                             thickness=1.5, font_size=18, win_name='', show=False, wait_time=0,
                             out_file=None, is_draw_bbox=False, is_draw_labels=False):
    img = mmcv.imread(img)
    img = img.copy()
    if isinstance(result, tuple):
        bbox_result, segm_result = result
        if isinstance(segm_result, tuple):
            segm_result = segm_result[0]  # ms rcnn
    else:
        bbox_result, segm_result = result, None
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    # draw segmentation masks
    segms = None
    if segm_result is not None and len(labels) > 0:  # non empty
        segms = mmcv.concat_list(segm_result)
        if isinstance(segms[0], torch.Tensor):
            segms = torch.stack(segms, dim=0).detach().cpu().numpy()
        else:
            segms = np.stack(segms, axis=0)
    # if out_file specified, do not show image in window
    if out_file is not None:
        show = False
    # draw bounding boxes
    img = imshow_det_bboxes(
        img,
        bboxes,
        labels,
        segms,
        prediction_map,
        class_names=CLASSES,
        score_thr=score_thr,
        bbox_color=bbox_color,
        text_color=text_color,
        mask_color=mask_color,
        thickness=thickness,
        font_size=font_size,
        win_name=win_name,
        show=show,
        wait_time=wait_time,
        out_file=out_file,
        is_draw_bbox=is_draw_bbox,
        is_draw_labels=is_draw_labels)

    if not (show or out_file):
        return img


def imshow_det_bboxes(img, bboxes=None, labels=None, segms=None, prediction_map=None, class_names=None,
                      score_thr=0, bbox_color='green', text_color='green', mask_color=None,
                      thickness=2, font_size=8, win_name='', show=True, wait_time=0,
                      out_file=None, is_draw_bbox=False, is_draw_labels=False):
    assert bboxes is None or bboxes.ndim == 2, \
        f' bboxes ndim should be 2, but its ndim is {bboxes.ndim}.'
    assert labels.ndim == 1, \
        f' labels ndim should be 1, but its ndim is {labels.ndim}.'
    assert bboxes is None or bboxes.shape[1] == 4 or bboxes.shape[1] == 5, \
        f' bboxes.shape[1] should be 4 or 5, but its {bboxes.shape[1]}.'
    assert bboxes is None or bboxes.shape[0] <= labels.shape[0], \
        'labels.shape[0] should not be less than bboxes.shape[0].'
    assert segms is None or segms.shape[0] == labels.shape[0], \
        'segms.shape[0] and labels.shape[0] should have the same length.'
    assert segms is not None or bboxes is not None, \
        'segms and bboxes should not be None at the same time.'

    img = mmcv.imread(img).astype(np.uint8)

    if score_thr > 0:
        assert bboxes is not None and bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]
        if segms is not None:
            segms = segms[inds, ...]

    img = mmcv.bgr2rgb(img)
    width, height = img.shape[1], img.shape[0]
    img = np.ascontiguousarray(img)

    fig = plt.figure(win_name, frameon=False)
    plt.title(win_name)
    canvas = fig.canvas
    dpi = fig.get_dpi()
    # add a small EPS to avoid precision lost due to matplotlib's truncation
    # (https://github.com/matplotlib/matplotlib/issues/15363)
    fig.set_size_inches((width + EPS) / dpi, (height + EPS) / dpi)

    # remove white edges by set subplot margin
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    ax = plt.gca()
    ax.axis('off')

    max_label = int(max(labels) if len(labels) > 0 else 0)
    text_palette = palette_val(get_palette(text_color, max_label + 1))
    text_colors = [text_palette[label] for label in labels]

    num_bboxes = 0
    if bboxes is not None and is_draw_bbox:
        num_bboxes = bboxes.shape[0]
        bbox_palette = palette_val(get_palette(bbox_color, max_label + 1))
        colors = [bbox_palette[label] for label in labels[:num_bboxes]]
        draw_bboxes(ax, bboxes, colors, alpha=0.8, thickness=thickness)

        horizontal_alignment = 'left'
        positions = bboxes[:, :2].astype(np.int32) + thickness
        areas = (bboxes[:, 3] - bboxes[:, 1]) * (bboxes[:, 2] - bboxes[:, 0])
        scales = _get_adaptive_scales(areas)
        scores = bboxes[:, 4] if bboxes.shape[1] == 5 else None
        if is_draw_labels:
            # Caculate mean depths
            segms_bool = segms.astype(bool) if segms is not None else np.zeros((len(labels), height, width), dtype=bool)  # (N, H, W)
            depth_maps = np.broadcast_to(prediction_map, segms_bool.shape)  # (N, H, W)
            sum_depths = (depth_maps * segms_bool).sum(axis=(1, 2))  # (N,)
            pixel_counts = np.maximum(segms_bool.sum(axis=(1, 2)), 1)  # (N,)
            mean_depths = sum_depths / pixel_counts  # (N,)

            draw_labels(
                ax,
                labels[:num_bboxes],
                positions,
                scores=scores,
                mean_depths=mean_depths,
                class_names=class_names,
                color=text_colors,
                font_size=font_size,
                scales=scales,
                horizontal_alignment=horizontal_alignment,
                is_draw_score=DRAW_SCORE,
                is_draw_depth=DRAW_DEPTHS)

    if segms is not None:
        mask_palette = get_palette(mask_color, max_label + 1)
        colors = [mask_palette[label] for label in labels]
        colors = np.array(colors, dtype=np.uint8)
        draw_masks(ax, img, segms, colors, with_edge=True)

        if num_bboxes < segms.shape[0]:
            segms = segms[num_bboxes:]
            horizontal_alignment = 'center'
            areas = []
            positions = []
            for mask in segms:
                _, _, stats, centroids = cv2.connectedComponentsWithStats(
                    mask.astype(np.uint8), connectivity=8)
                largest_id = np.argmax(stats[1:, -1]) + 1
                positions.append(centroids[largest_id])
                areas.append(stats[largest_id, -1])
            areas = np.stack(areas, axis=0)
            scales = _get_adaptive_scales(areas)
            if is_draw_labels:
                # Caculate mean depths
                segms_bool = segms.astype(bool)  # (N, H, W)
                depth_maps = np.broadcast_to(prediction_map, segms.shape)  # (N, H, W)
                sum_depths = (depth_maps * segms_bool).sum(axis=(1, 2))  # (N,)
                pixel_counts = np.maximum(segms_bool.sum(axis=(1, 2)), 1)  # (N,)
                mean_depths = sum_depths / pixel_counts  # (N,)

                draw_labels(
                    ax,
                    labels[num_bboxes:],
                    positions,
                    mean_depths=mean_depths,
                    class_names=class_names,
                    color=text_colors,
                    font_size=font_size,
                    scales=scales,
                    horizontal_alignment=horizontal_alignment)

    plt.imshow(img)

    stream, _ = canvas.print_to_buffer()
    buffer = np.frombuffer(stream, dtype='uint8')
    if sys.platform == 'darwin':
        width, height = canvas.get_width_height(physical=True)
    img_rgba = buffer.reshape(height, width, 4)
    rgb, alpha = np.split(img_rgba, [3], axis=2)
    img = rgb.astype('uint8')
    img = mmcv.rgb2bgr(img)

    if show:
        # We do not use cv2 for display because in some cases, opencv will
        # conflict with Qt, it will output a warning: Current thread
        # is not the object's thread. You can refer to
        # https://github.com/opencv/opencv-python/issues/46 for details
        if wait_time == 0:
            plt.show()
        else:
            plt.show(block=False)
            plt.pause(wait_time)
    if out_file is not None:
        mmcv.imwrite(img, out_file)

    plt.close()

    return img


def _get_adaptive_scales(areas, min_area=800, max_area=30000):
    """Get adaptive scales according to areas.

    The scale range is [0.5, 1.0]. When the area is less than
    ``'min_area'``, the scale is 0.5 while the area is larger than
    ``'max_area'``, the scale is 1.0.

    Args:
        areas (ndarray): The areas of bboxes or masks with the
            shape of (n, ).
        min_area (int): Lower bound areas for adaptive scales.
            Default: 800.
        max_area (int): Upper bound areas for adaptive scales.
            Default: 30000.

    Returns:
        ndarray: The adaotive scales with the shape of (n, ).
    """
    scales = 0.5 + (areas - min_area) / (max_area - min_area)
    scales = np.clip(scales, 0.5, 1.0)
    return scales


def _get_bias_color(base, max_dist=30):
    """Get different colors for each masks.

    Get different colors for each masks by adding a bias
    color to the base category color.
    Args:
        base (ndarray): The base category color with the shape
            of (3, ).
        max_dist (int): The max distance of bias. Default: 30.

    Returns:
        ndarray: The new color for a mask with the shape of (3, ).
    """
    new_color = base + np.random.randint(
        low=-max_dist, high=max_dist + 1, size=3)
    return np.clip(new_color, 0, 255, new_color)


def draw_bboxes(ax, bboxes, color='g', alpha=0.8, thickness=2):
    """Draw bounding boxes on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        bboxes (ndarray): The input bounding boxes with the shape
            of (n, 4).
        color (list[tuple] | matplotlib.color): the colors for each
            bounding boxes.
        alpha (float): Transparency of bounding boxes. Default: 0.8.
        thickness (int): Thickness of lines. Default: 2.

    Returns:
        matplotlib.Axes: The result axes.
    """
    polygons = []
    for i, bbox in enumerate(bboxes):
        bbox_int = bbox.astype(np.int32)
        poly = [[bbox_int[0], bbox_int[1]], [bbox_int[0], bbox_int[3]],
                [bbox_int[2], bbox_int[3]], [bbox_int[2], bbox_int[1]]]
        np_poly = np.array(poly).reshape((4, 2))
        polygons.append(Polygon(np_poly))
    p = PatchCollection(
        polygons,
        facecolor='none',
        edgecolors=color,
        linewidths=thickness,
        alpha=alpha)
    ax.add_collection(p)

    return ax


def draw_labels(ax,
                labels,
                positions,
                scores=None,
                mean_depths=None,
                class_names=None,
                color='w',
                font_size=8,
                scales=None,
                horizontal_alignment='left',
                is_draw_score=True,
                is_draw_depth=True):
    """Draw labels on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        labels (ndarray): The labels with the shape of (n, ).
        positions (ndarray): The positions to draw each labels.
        scores (ndarray): The scores for each labels.
        class_names (list[str]): The class names.
        color (list[tuple] | matplotlib.color): The colors for labels.
        font_size (int): Font size of texts. Default: 8.
        scales (list[float]): Scales of texts. Default: None.
        horizontal_alignment (str): The horizontal alignment method of
            texts. Default: 'left'.

    Returns:
        matplotlib.Axes: The result axes.
    """
    for i, (pos, label) in enumerate(zip(positions, labels)):
        label_text = class_names[
            label] if class_names is not None else f'class {label}'
        # Draw mean depth
        if mean_depths is not None and is_draw_depth:
            label_text += f'|{mean_depths[i]:.02f} m'
        # Draw score
        if scores is not None and is_draw_score:
            label_text += f'|{scores[i]:.02f}'
        text_color = color[i] if isinstance(color, list) else color

        font_size_mask = font_size if scales is None else font_size * scales[i]
        ax.text(
            pos[0],
            pos[1] - int(font_size_mask * 1.7),
            f'{label_text}',
            bbox={
                'facecolor': 'black',
                'alpha': 0.8,
                'pad': 0.7,
                'edgecolor': 'none'
            },
            color=text_color,
            fontsize=font_size_mask,
            verticalalignment='top',
            horizontalalignment=horizontal_alignment)

    return ax


def draw_masks(ax, img, masks, color=None, with_edge=True, alpha=0.8):
    """Draw masks on the image and their edges on the axes.

    Args:
        ax (matplotlib.Axes): The input axes.
        img (ndarray): The image with the shape of (3, h, w).
        masks (ndarray): The masks with the shape of (n, h, w).
        color (ndarray): The colors for each masks with the shape
            of (n, 3).
        with_edge (bool): Whether to draw edges. Default: True.
        alpha (float): Transparency of bounding boxes. Default: 0.8.

    Returns:
        matplotlib.Axes: The result axes.
        ndarray: The result image.
    """
    taken_colors = set([0, 0, 0])
    if color is None:
        random_colors = np.random.randint(0, 255, (masks.size(0), 3))
        color = [tuple(c) for c in random_colors]
        color = np.array(color, dtype=np.uint8)
    polygons = []
    for i, mask in enumerate(masks):
        if with_edge:
            contours, _ = bitmap_to_polygon(mask)
            polygons += [Polygon(c) for c in contours]

        color_mask = color[i]
        while tuple(color_mask) in taken_colors:
            color_mask = _get_bias_color(color_mask)
        taken_colors.add(tuple(color_mask))

        mask = mask.astype(bool)
        img[mask] = img[mask] * (1 - alpha) + color_mask * alpha

    p = PatchCollection(
        polygons, facecolor='none', edgecolors='w', linewidths=1, alpha=0.8)
    ax.add_collection(p)

    return ax, img


@torch.no_grad()
def inference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_file", type=str, default="./outputs/inference.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs/vis_depth_segms")
    parser.add_argument("--model", type=str, default="Swin", choices=["Swin", "ConvNeXt"])
    args = parser.parse_args()

    SEG_MODEL_CONFIG = SEG_SWIN_MODEL_CONFIG if args.model == "Swin" else SEG_CONVNEXT_MODEL_CONFIG
    SEG_MODEL_CKPT = SEG_SWIN_MODEL_CKPT if args.model == "Swin" else SEG_CONVNEXT_MODEL_CKPT

    DATASET = get_example_dataset_inference(index_file=args.index_file)
    path_tuples = DATASET.path_tuples

    OUT_PATH = args.output_dir
    os.makedirs(OUT_PATH, exist_ok=True)

    print(f"Using device {DEVICE}")

    # Load models
    depth_model = UDFNet(n_bins=80).to(DEVICE)
    depth_model.load_state_dict(torch.load(DEPTH_MODEL_PATH, map_location=DEVICE))
    depth_model.eval()

    seg_model = init_detector(SEG_MODEL_CONFIG, SEG_MODEL_CKPT, device=DEVICE)

    dataloader = DataLoader(DATASET, batch_size=BATCH_SIZE, drop_last=True)
    total_time_per_image = 0.0

    for batch_id, data in enumerate(tqdm(dataloader)):
        rgb = data[0].to(DEVICE)
        prior = data[1].to(DEVICE)
        segms_input = data[2].to(DEVICE)

        start_time = time.time()
        prediction, _ = depth_model(rgb, prior, segms_input)
        end_time = time.time()

        total_time_per_image += (end_time - start_time) / rgb.size(0)

        for i in range(rgb.size(0)):
            index = batch_id * BATCH_SIZE + i
            rgb_path = path_tuples[index][0]
            out_filename = splitext(basename(rgb_path))[0]
            img = mmcv.imread(rgb_path)
            height, width = img.shape[:2]

            prediction_map = F.interpolate(
                prediction[i].unsqueeze(0), size=(height, width),
                mode="bilinear", align_corners=False
            ).squeeze(0).cpu().numpy()

            # Run segmentation
            result = inference_detector(seg_model, rgb_path)

            out_file = join(OUT_PATH, f"{out_filename}_objdepth_vis.png")

            # Show Depth-Inform Instance Segmentation
            show_depth_segms_results(
                img=img,
                result=result,
                prediction_map=prediction_map,
                score_thr=SCORE_THR,
                out_file=out_file,
                show=True,
                wait_time=0,
                win_name='result',
                bbox_color=PALETTE,
                text_color=(200, 200, 200),
                mask_color=PALETTE,
                is_draw_bbox=DRAW_BBOX,
                is_draw_labels=DRAW_LABELS)

    avg_time = total_time_per_image / len(dataloader)
    print(f"Average time per image: {avg_time:.4f}s, FPS: {1.0 / avg_time:.2f}")


if __name__ == "__main__":
    inference()
