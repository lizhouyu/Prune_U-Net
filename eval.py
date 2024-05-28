import torch
import torch.nn as nn

from tqdm import tqdm
import numpy as np
from dice_loss import dice_coeff
from utils import batch

# IoU score is calculated as intersection / union
def calculate_iou(gt, pred):
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

# Dice score is calculated as 2 * intersection / (sum of gt and pred)
def calculate_dice(gt, pred):
    intersection = np.logical_and(gt, pred)
    sum_gt_pred = np.sum(gt) + np.sum(pred)
    if sum_gt_pred == 0:
        return 1
    dice_score = 2 * np.sum(intersection) / sum_gt_pred
    return dice_score

def eval_net(net, dataloader, gpu=False):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    tot = 0
    with torch.no_grad(), tqdm(total=len(dataloader)) as progress_bar:
        for batch_idx, (imgs, true_masks) in enumerate(dataloader):
            if gpu and torch.cuda.is_available():
                imgs = imgs.cuda()

            masks_pred = net(imgs)[0]
            masks_pred = (masks_pred > 0.5).float()
            # get data back to numpy
            masks_pred = masks_pred.cpu().numpy()
            true_masks = true_masks[0].numpy()
            # calculate dice score
            dice = calculate_dice(masks_pred, true_masks)
            tot += dice
            progress_bar.set_postfix(DICE=dice)
            progress_bar.update(1)
    value = tot / len(dataloader.dataset)
    return value

if __name__ == "__main__":
    # make two random 0-1 masks
    mask = np.random.randint(0, 2, (1, 320, 320))
    im = np.random.randint(0, 2, (1, 320, 320))
    dice = calculate_dice(im, mask)
    print("dice score", dice)
    dice_same = calculate_dice(im, im)
    print("dice score for same image", dice_same)
    dice_diff = calculate_dice(im, 1 - im)
    print("dice score for different image", dice_diff)
