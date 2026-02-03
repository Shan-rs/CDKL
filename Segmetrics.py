#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 10:15:35 2023

@author: ws-512
"""

from utils import stream_metrics
import glob
import numpy as np
import cv2
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as T

class ThresholdTransform(object):
    def __init__(self, th=0.5):
        self.th = th
    def __call__(self, x: torch.Tensor):
        # x: [C,H,W] in [0,1] float32
        return (x >= self.th).long()  # [C,H,W] int64

transform_label = T.Compose([
    T.Resize((256, 256)),
    T.ToTensor(),                 # -> [0,1] float
    ThresholdTransform(th=0.9),   # 你给的阈值
])

transform_pred = T.Compose([
    T.ToTensor(),                 # -> [0,1] float
    ThresholdTransform(th=0.5),   # 预测阈值可按需要调整
])

if __name__ == '__main__':
    
    predict_dir = glob.glob('/home/user/H/ObjectDetection2023_revised/Snet_DK2_0.001/*')
    label_path = '/home/user/H/remote_sensing_datasets/OurUnuniformHazeDataset/test/gt/'
    metrics = stream_metrics.StreamSegMetrics(2)
    metrics.reset()

    for predict_path in predict_dir:
        # --- 读预测 ---
        pred_img = Image.open(predict_path).convert('L')
        pred_t   = transform_pred(pred_img)              # [1,256,256], int64 0/1
        # 如果 metrics 期望 numpy，则转为 [H,W] numpy
        pred_np  = pred_t.squeeze(0).cpu().numpy()       # [256,256], {0,1}

        # --- 找对应标签名 ---
        predict_name = predict_path.split('/')[-1]
        label_name   = predict_name[:-6] + '.jpg'        # 根据你之前的命名规则

        # --- 读标签 ---
        lab_img = Image.open(label_path + label_name).convert('L')
        lab_t   = transform_label(lab_img)               # [1,256,256], int64 0/1
        lab_np  = lab_t.squeeze(0).cpu().numpy()         # [256,256], {0,1}

        # --- 更新指标 ---
        metrics.update(lab_np, pred_np)

    score = metrics.get_results()
    print(metrics.to_str(score))

    # for predict_path in predict_dir:
    #     imgPredict = cv2.imread(predict_path)
    #     predict_name = predict_path.split('/')[-1]
    #     label_name = predict_name
    #     label_name = predict_name[:-6] + '.jpg'
    #     imgLabel = cv2.imread(label_path + label_name)
    #     imgLabel = cv2.resize(imgLabel, (256, 256))
        
    #     imgPredict = np.array(cv2.cvtColor(imgPredict, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
    #     imgLabel = np.array(cv2.cvtColor(imgLabel, cv2.COLOR_BGR2GRAY) / 255., dtype=np.uint8)
        
    #     metrics.update(imgLabel, imgPredict)
    # score = metrics.get_results()
    # list1 = [key for key in score.keys()]
        
    # print(metrics.to_str(score))


