import torch
import torch.nn as nn
import torch.onnx

from unet import UNet

# check device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("device: ", device)

# set model path 
# model_path = 'checkpoints/best.pt'
model_path = 'checkpoints/prune/unet_pruned_l1_80.pt'

# set output path
# output_path = 'unet.onnx'
output_path = 'unet_pruned_l1_80.onnx'

# load model
model = UNet(in_channels=3, out_channels=1)
model.load_state_dict(torch.load(model_path))
model.to(device)

# prepare arguments
dummy_input = torch.randn(1, 3, 320, 320, device=device)
input_names = ["in"]
output_names = ["out"]
dynamic_axes = {'in': {0: 'batch'}, 'out': {0: 'batch'}}

torch.onnx.export(model, dummy_input, output_path, verbose=False, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)
