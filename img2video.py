import math
import cv2
import os
import argparse
from natsort import natsorted
import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Convert images to video')
    parser.add_argument('--video', help='video path')
    parser.add_argument('--input', type=str, default='./processed_video', help='input image folder')
    parser.add_argument('--result', type=str, default='./outputs/vis_depth_segms', help='result image folder')
    parser.add_argument('--output', type=str, default='./outputs', help='output video folder')
    parser.add_argument('--frame_interval', type=int, default=10, help='frame interval')
    parser.add_argument('--ext', type=str, default='png', help='image file extension (e.g., png, jpg)')
    return parser.parse_args()


def images_to_video(input_folder, output_folder, fps=12, ext='png', output_name="outputs.mp4"):
    os.makedirs(output_folder, exist_ok=True)
    video_filename = os.path.join(output_folder, output_name)

    image_files = [f for f in os.listdir(input_folder) if f.endswith(f".{ext}")]
    image_files = natsorted(image_files)

    if not image_files:
        print("No images found in", input_folder)
        return

    first_image_path = os.path.join(input_folder, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print("Failed to read image:", first_image_path)
        return

    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(video_filename, fourcc, fps, (width, height))

    for i, img_file in enumerate(image_files):
        img_path = os.path.join(input_folder, img_file)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: failed to read {img_path}, skipping.")
            continue

        writer.write(frame)
        print(f"\rWriting frame {i+1}/{len(image_files)}", end='')

    writer.release()
    print(f"\nVideo saved to {video_filename}")


def main():
    args = parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Unable to open the video:", args.video)
        return
    ori_fps = math.ceil(cap.get(cv2.CAP_PROP_FPS))
    output_fps = ori_fps // args.frame_interval
    print('ori_fps    : ', ori_fps)
    print('output_fps : ', output_fps)

    images_to_video(
        input_folder=args.input,
        output_folder=args.output,
        output_name="input.mp4",
        fps=output_fps,
        ext=args.ext
    )
    images_to_video(
        input_folder=args.result,
        output_folder=args.output,
        output_name="results.mp4",
        fps=output_fps,
        ext=args.ext
    )


if __name__ == '__main__':
    main()
