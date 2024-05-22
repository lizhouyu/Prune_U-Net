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
    parser = argparse.ArgumentParser(description='Infer with U-Net')
    parser.add_argument('--model', type=str, default='checkpoints/best.pt', help='path to model')
    parser.add_argument('--input', type=str, default='subscene.png', help='path to input image')
    parser.add_argument('--output', type=str, default='predict.png', help='path to output image')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = load_model(args.model)
    
    input = cv2.imread(args.input)
    input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
    # input = np.expand_dims(input, axis=0)
    # change 1x320x320x3 to 1x3x320x320
    # input = np.transpose(input, (0, 3, 1, 2))
    input = transforms.ToTensor()(input)
    input = input.unsqueeze(0)
    print(input.shape)
    # get the max, min values of the input tensor
    max_val = input.max()
    min_val = input.min()
    print(max_val, min_val)
    # convert the permuted tensor to numpy and save it as an image
    input_to_save = input.squeeze().detach().numpy()
    input_to_save = input_to_save.astype(np.uint8)
    print(input_to_save.shape)
    # permute back to 320x320x3
    input_to_save = np.transpose(input_to_save, (1, 2, 0))
    input_to_save = cv2.cvtColor(input_to_save, cv2.COLOR_RGB2BGR)
    cv2.imwrite('input.png', input_to_save)
    output = infer(model, input)

    output = output.squeeze().detach().numpy()
    
    # save the output values as a txt file, each value keeps 2 decimal places
    np.savetxt('output.txt', output, fmt='%.2f')

    # from output numpy array compute the model's prediction entropy of the image 
    # (1/num_pixels)*\sum_{each pixel} -plogp where p is the probability of the pixel given in the numpy array
    w, h = output.shape
    entropy = (-1/(w*h))*np.sum(output*np.log(output))
    print(entropy)

    # get an array of all 0.5 with the same shape as the output
    x = np.full((w, h), 0.5)
    print(x)
    print(- np.sum(x * np.log(x) / (w * h)))

    # output a mask with threshold 0.5
    output = (output > 0.5) * 255
    output = output.astype(np.uint8)
    output = Image.fromarray(output)
    output.save(args.output)