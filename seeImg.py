import os
import cv2
import argparse
import torch
from lib.utils.demo_utils import (
    download_youtube_clip,
    smplify_runner,
    convert_crop_coords_to_orig_img,
    convert_crop_cam_to_orig_img,
    prepare_rendering_results,
    video_to_images,
    images_to_video,
    download_ckpt,
)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    video_file = args.vid_file

    if not os.path.isfile(video_file):
        exit(f'Input video \"{video_file}\" does not exist!')
    output_path = os.path.join(args.output_folder, os.path.basename(video_file).replace('.mp4', ''))
    os.makedirs(output_path, exist_ok=True)
    image_folder, num_frames, img_shape = video_to_images(video_file, return_info=True)
    original_image_path = os.path.join(image_folder, '000001.png')
    original_image = cv2.imread(original_image_path)
    cv2.imshow('Original Image', original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str, required=True, help='Path to the video file')
    parser.add_argument('--output_folder', type=str, required=True, help='Folder to save the output')

    args = parser.parse_args()
    main(args)
