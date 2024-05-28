import sys
import os
import cv2
import json
import argparse
import numpy as np
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torchvision import transforms

from eval import eval_net
from unet import UNet
from utils.dataset import UNetDataset

def infer(model: UNet, input: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(input)
    return output

def load_model(model_path: str) -> UNet:
    model = UNet(in_channels=3, out_channels=1)
    model.load_state_dict(torch.load(model_path))
    return model

if __name__ == '__main__':

    # val_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/subscenes'
    # val_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/masks'

    val_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/eval_data_drift/data/subscenes'
    val_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/eval_data_drift/data/masks'

    val_dataset = UNetDataset(im_folder_path=val_im_folder_path, mask_folder_path=val_mask_folder_path, format='image')
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = load_model('checkpoints/best.pt')
    if torch.cuda.is_available():
        model.cuda()
    
    val_dice = eval_net(model, val_dataloader, True)

    print("Validation Dice Coefficient: {}".format(val_dice))