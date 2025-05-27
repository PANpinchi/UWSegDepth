import cv2
import os
import sys
import argparse

import warnings
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing Video')
    parser.add_argument('--video', help='video path')
    parser.add_argument('--frame_interval', type=int, default=10, help='frame interval')
    parser.add_argument('--output', type=str, default='processed_video', help='output folder')
    args = parser.parse_args()
    return args


def video_to_images(video_path, output_folder, frame_interval=5):
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open the video:", video_path)
        return

    frame_count = 0
    saved_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"{frame_count:04d}.png")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1

        progress = (frame_count / total_frames) * 100
        sys.stdout.write(f"\rProcessing progress: {progress:.2f}% ({frame_count}/{total_frames} frames)")
        sys.stdout.flush()

        frame_count += 1

    cap.release()
    print(f"\nConversion completed, {saved_count} images stored in {output_folder}")


def main():
    args = parse_args()
    video_to_images(video_path=args.video, output_folder=args.output, frame_interval=args.frame_interval)


if __name__ == '__main__':
    main()