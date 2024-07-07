import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from eval import eval_net
from unet import UNet
from utils.dataset import UNetDataset

from tqdm import tqdm

def train_net(net,
              epochs=5,
              batch_size=1,
              lr=0.1,
              val_percent=0.05,
              save_cp=True,
              gpu=False,
              img_scale=0.5):
    
    # seed random generator for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    train_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/train/subscenes'
    train_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/train/masks'
    val_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/subscenes'
    val_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/masks'

    dir_checkpoint = 'checkpoints/original/'
    os.makedirs(dir_checkpoint, exist_ok=True)

    train_dataset = UNetDataset(im_folder_path=train_im_folder_path, mask_folder_path=train_mask_folder_path, format='image')
    val_dataset = UNetDataset(im_folder_path=val_im_folder_path, mask_folder_path=val_mask_folder_path, format='image')

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(train_dataset),
               len(val_dataset), str(save_cp), str(gpu)))

    N_train = len(train_dataset)

    optimizer = optim.Adam(net.parameters(), lr=lr, eps=9.99999993923e-09, betas=(0.899999976158, 0.999000012875))

    criterion = nn.BCELoss()
    l2_reg_func = nn.MSELoss()

    val_dice_max = 0

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()

        # reset the generators
        # train = get_imgs_and_masks(iddataset['train'], dir_img, dir_mask, img_scale)
        # val = get_imgs_and_masks(iddataset['val'], dir_img, dir_mask, img_scale)


        epoch_loss = 0

        pbar = tqdm(train_dataloader)

        for batch_idx, (imgs, true_masks) in enumerate(pbar):
            if gpu:
                imgs = imgs.cuda()
                true_masks = true_masks.cuda()

            masks_pred = net(imgs)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = true_masks.view(-1)

            optimizer.zero_grad()

            cse_loss = criterion(masks_probs_flat, true_masks_flat)
            l2_reg = l2_reg_func(masks_probs_flat, true_masks_flat)
            loss = cse_loss + l2_reg * 2e-5
            epoch_loss += loss.item()

            # if batch_idx % 50 == 0:
            #     print('{0:.4f} --- loss: {1:.6f}'.format(batch_idx * batch_size / N_train, loss.item() / true_masks.size(0)))
            

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=loss.item() / true_masks.size(0))
            pbar.update(1)

        print('Epoch finished ! Loss: {}'.format(epoch_loss / len(train_dataset)))

        val_dice = eval_net(net, val_dataloader, gpu)
        print('Validation Dice Coeff: {}'.format(val_dice))

        if save_cp:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'last.pt'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))
            
        if val_dice > val_dice_max:
            val_dice_max = val_dice
            torch.save(net.state_dict(),
                       dir_checkpoint + 'best.pt')
            print('Best Checkpoint saved !')



def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=50, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.0001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=3, n_classes=1, f_channels='model_channels.txt')
    # net = UNet(in_channels=3, out_channels=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        if torch.cuda.is_available():
            net.cuda()
        else:
            print('Cuda is not available, using CPU')
            args.gpu = False
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  gpu=args.gpu,
                  img_scale=args.scale)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pt')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
