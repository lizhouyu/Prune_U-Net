import torch
import numpy as np
from torch.utils.data import DataLoader

from tqdm import tqdm
from utils import batch, AverageMeter, get_imgs_and_masks
from flops_counter import flops_count


def finetune(net, optimizer, criterion, l2_reg_func, trainset, log, path, iters=100, epochs=None, batch_size=2, gpu=True, scale=0.5):
    net.train()
    bce_meter = AverageMeter()

    dir_img = 'data/train/'
    dir_mask = 'data/train_masks/'

    if epochs is None:  # Fine-tune using iterations of mini-batches
        epochs = 1
    else:  # Fine-tune using entire epochs
        iters = None

    for e in range(epochs):
        # reset the generators
        # train = get_imgs_and_masks(trainset, dir_img, dir_mask, scale)
        train_dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        with tqdm(total=len(train_dataloader)) as progress_bar:
            for batch_idx, (imgs, true_masks) in enumerate(train_dataloader):
                if gpu:
                    imgs = imgs.cuda()
                    true_masks = true_masks.cuda()

                masks_pred = net(imgs)

                cse_loss = criterion(masks_pred, true_masks)
                l2_reg = l2_reg_func(masks_pred, true_masks)
                loss = cse_loss + l2_reg * 2e-5

                bce_meter.update(loss.item(), batch_size)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                progress_bar.update(batch_size)
                progress_bar.set_postfix(epoch=e, BCE=bce_meter.avg)

                if batch_idx == 0 and e == 0:
                    log.info("FLOPs after pruning: \n{}".format(flops_count(net, imgs.shape[2:])))

                if batch_idx == iters:  # Stop finetuning after sufficient mini-batches
                    break

    log.info("Finished finetuning")
    log.info("Finetuned loss: {}".format(bce_meter.avg))
    torch.save(net.state_dict(), path)
    log.info('Saving finetuned to {}...'.format(path))