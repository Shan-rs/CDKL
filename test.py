import glob
import math
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from PIL import Image
from thop import profile
from torchvision import transforms as T

from config import Config
import utils.save_load as sl
from T_Model import T_network
from S_Model import S_network

def save_cam_colormap(cam_2d, save_path, colormap=cv2.COLORMAP_JET):
    """
    Saves a 2D attention map/Grad-CAM as a pseudo-color heatmap.
    cam_2d: torch.Tensor or np.ndarray (H, W).
    """
    if hasattr(cam_2d, "detach"):
        cam_2d = cam_2d.detach().cpu().numpy()
    cam = cam_2d - cam_2d.min()
    cam = cam / (cam.max() + 1e-6)
    cam_u8 = np.uint8(cam * 255)
    heat_bgr = cv2.applyColorMap(cam_u8, colormap)
    cv2.imwrite(save_path, heat_bgr)


def normPRED(d):
    ma = torch.max(d)
    mi = torch.min(d)
    dn = (d-mi)/(ma-mi)
    return dn

def mytest(opt, imgs, selctnet, outgt_path, outclear_path):
    transform = T.Compose([T.Resize(256), T.ToTensor()])
    
    if not os.path.exists(outgt_path):
        os.mkdir(outgt_path)
    
    if selctnet == 'T-net':
        model = T_network()
        model, _, _, _ = sl.load_state(opt.load_Tmodel_path, model)
        if torch.cuda.is_available():
            model = model.cuda().eval()

        for img in imgs:
            hazy = Image.open(img)
            output_name = img.split('/')[-1]
            hazy = transform(hazy).unsqueeze(0)
            if torch.cuda.is_available():
                hazy = hazy.cuda()

            with torch.no_grad():
                # Extract classification/segmentation output
                _, _, _, out, _, _, _, _ = model(hazy)

            pred = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
            torchvision.utils.save_image(pred, os.path.join(outgt_path, output_name))

    if selctnet == 'S-net':
        model = S_network()
        model, _, _, _ = sl.load_state(opt.load_model_path, model)
        if torch.cuda.is_available():
            model = model.cuda().eval()

        if not os.path.exists(outclear_path):
            os.mkdir(outclear_path)

        for img in imgs:
            hazy = Image.open(img)
            output_name = os.path.basename(img)
            hazy = transform(hazy).unsqueeze(0)
            if torch.cuda.is_available():
                hazy = hazy.cuda()

            with torch.no_grad():
                _, _, S_dehazeout, _, _, _, out, _, _, _, _ = model(hazy)

            pred = torch.where(out > 0.5, torch.ones_like(out), torch.zeros_like(out))
            torchvision.utils.save_image(pred, os.path.join(outgt_path, output_name))

    # Calculate FLOPs and parameters
    flops, params = profile(model, inputs=(hazy,))
    print(f"Total FLOPs: {flops}")
    print(f"Total Parameters: {params}")


if __name__ == '__main__':
    opt = Config()
    # Path to test hazy images
    imgs = glob.glob('')
    selectnet = 'S-net'
    outgt_path = './Snet_results'
    outclear_path = 'dehazed_results'

    start_time = time.time()
    mytest(opt, imgs, selectnet, outgt_path, outclear_path)
    end_time = time.time()

    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("Inference completed!")
