import os
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader

from unet import UNet
from utils.dataset import UNetDataset 
from eval import eval_net

# set the input file path
input_file_path = "checkpoints/best.pt"
assert os.path.exists(input_file_path), f"Invalid input file path: {input_file_path}"

# set the output file path
output_folder_path = "checkpoints/prune"
output_file_path = os.path.join(output_folder_path, 'unet_pruned_l1_90.pt')
os.makedirs(output_folder_path, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate the model
model = UNet(in_channels=3, out_channels=1)
# load pre-trained weights
model.load_state_dict(torch.load(input_file_path))
# move the model to the device
model.to(device)

# Dataset
val_im_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/subscenes'
val_mask_folder_path = '/mnt/data/zli85/github/OECModelRetraining/pilot_study/data_drift_existence/cloud_detection/unet/train_tao_unet_cloud/tao_experiments/data/320/val/masks'
val_dataset = UNetDataset(im_folder_path=val_im_folder_path, mask_folder_path=val_mask_folder_path, format='image')
val_dataloader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)

# evaluate the model before pruning
val_score = eval_net(model, val_dataloader, gpu=True if device.type == 'cuda' else False)
print(f"Validation Dice Coeff before pruning: {val_score}")

# add parameters to be pruned
parameters_to_prune = []

for name, module in model.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        if 'conv1' in name or 'convTranspose' in name:
            parameters_to_prune.append((module, 'weight'))
            # print the sparsity of the module
            print(f"Sparsity in {name} before pruning: {torch.sum(module.weight == 0).item() / module.weight.nelement()}")

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured, # L1 pruning
    amount=0.9, # 80% sparsity
)

print("Pruning complete")

# remove pruning re-parametrization
for module, _ in parameters_to_prune:
    prune.remove(module, 'weight')

# evaluate the model after pruning
val_score = eval_net(model, val_dataloader, gpu=True if device.type == 'cuda' else False)
print(f"Validation Dice Coeff after pruning: {val_score}")

# save the pruned model
torch.save(model.state_dict(), output_file_path)

# try to load the pruned model
model_pruned_load = UNet(in_channels=3, out_channels=1)
model_pruned_load.load_state_dict(torch.load(output_file_path))
model_pruned_load.to(device)
print("Pruned model loaded successfully")

# check the sparsity of the pruned model
for name, module in model_pruned_load.named_modules():
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        if 'conv1' in name or 'convTranspose' in name:
            print(f"Sparsity in {name}: {torch.sum(module.weight == 0).item() / module.weight.nelement()}")

# evaluate the pruned model
val_score = eval_net(model_pruned_load, val_dataloader, gpu=True if device.type == 'cuda' else False)
print(f"Validation Dice Coeff on loaded pruned model: {val_score}")


