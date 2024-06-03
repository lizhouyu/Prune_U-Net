import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
import os

from unet import UNet
from utils.dataset import UNetDataset 


from finetune import finetune
from eval import eval_net
from utils import get_logger

train_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/train/subscenes'
train_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/train/masks'
val_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/subscenes'
val_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/masks'
dir_checkpoint = 'checkpoints/finetune'
os.makedirs(dir_checkpoint, exist_ok=True)
finetuned_model_weight_path = 'checkpoints/finetune/unet_pruned_l1_30_finetuned.pt'

pruned_model_weight_path = 'checkpoints/prune/unet_pruned_l1_30.pt'

# get logger
log = get_logger(dir_checkpoint, 'finetune')  # logger

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate the model
model = UNet(in_channels=3, out_channels=1)
# load pre-trained weights
model.load_state_dict(torch.load(pruned_model_weight_path))
# move the model to the device
model.to(device)

# Dataset
train_dataset = UNetDataset(im_folder_path=train_im_folder_path, mask_folder_path=train_mask_folder_path, format='image')
val_dataset = UNetDataset(im_folder_path=val_im_folder_path, mask_folder_path=val_mask_folder_path, format='image')

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=4)
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# evaluate the model before fine-tuning
val_score = eval_net(model, val_dataloader, gpu=True if device.type == 'cuda' else False)
log.info(f"Validation Dice Coeff before fine-tuning: {val_score}")

# Loss function
criterion = nn.BCELoss()
# L2 regularization function
l2_reg_func = nn.MSELoss()

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.0001, eps=9.99999993923e-09, betas=(0.899999976158, 0.999000012875))

# Fine-tune the model
finetune(model, optimizer, criterion, l2_reg_func, train_dataset, log, path=finetuned_model_weight_path, iters=100, epochs=None, batch_size=2, gpu=True if device.type == 'cuda' else False)

val_score = eval_net(model, val_dataloader, gpu=True if device.type == 'cuda' else False)

log.info(f"Validation Dice Coeff: {val_score}")

