# demo.py (animated cartoon rigged to SMPL)
import os
import cv2
import torch
import argparse
import numpy as np
from cartoon_smpl import CartoonSMPL   # your custom rigged model
from renderer import Renderer

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load video
    cap = cv2.VideoCapture(args.vid_file)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Prepare output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output_file, fourcc, fps, (width, height))

    # Load cartoon SMPL (same API as SMPL, but your geometry)
    cartoon_smpl = CartoonSMPL(model_path=args.cartoon_model).to(device)

    renderer = Renderer(faces=cartoon_smpl.faces)

    frame_idx = 0
    while True:
        ret, img = cap.read()
        if not ret:
            break

        # get pose, betas, trans from VIBE (pretend we loaded them here)
        frame_pose = torch.zeros(1, 72).to(device)   # (replace with VIBE output)
        frame_betas = torch.zeros(1, 10).to(device)
        frame_trans = torch.zeros(1, 3).to(device)

        # deform your cartoon character
        output = cartoon_smpl(pose=frame_pose, betas=frame_betas, trans=frame_trans)
        frame_verts = output.vertices[0]

        rend_img = renderer.render(
            img,
            frame_verts,
            cam=np.array([1, 0, 0]),  # use real VIBE camera here
            color=[0.7, 0.3, 0.3]
        )

        out.write(rend_img)
        frame_idx += 1

    cap.release()
    out.release()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vid_file', type=str, required=True)
    parser.add_argument('--cartoon_model', type=str, required=True,
                        help='Your rigged cartoon SMPL-like model file')
    parser.add_argument('--output_file', type=str, default='output_cartoon.mp4')
    args = parser.parse_args()
    main(args)
