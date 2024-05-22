import onnx
import onnxruntime
import numpy as np
import cv2
import tensorrt

subscene_image_path = "subscene.png"
mask_image_path = "mask.png"

# Load the image
subscene_image = cv2.imread(subscene_image_path, cv2.IMREAD_COLOR)
mask_image = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)
subscene_image = np.array(subscene_image, dtype=np.float32)
mask_image = np.array(mask_image, dtype=np.float32)
# convert HWC to CHW
subscene_image = np.transpose(subscene_image, (2, 0, 1))
# unsqueeze to add batch dimension
subscene_image = np.expand_dims(subscene_image, axis=0)

# Load the ONNX model
model = onnx.load("model_rgb_320_pruned_retrained.onnx")

# Check that the IR is well formed
onnx.checker.check_model(model)

# Print a human readable representation of the graph
print(onnx.helper.printable_graph(model.graph))

# Run the model with ONNX Runtime
session = onnxruntime.InferenceSession(model.SerializeToString())

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print("Input name  :", input_name)
print("Output name :", output_name)



result = session.run([output_name], {input_name: subscene_image})
# get max and min values of the output
print("Max value: ", np.max(result))
print("Min value: ", np.min(result))

# get the layer before the output
second_from_last_layer = model.graph.node[-3].output[0]
print("Layer before output: ", second_from_last_layer)

# get the output of the layer before the output
result = session.run([second_from_last_layer], {input_name: subscene_image})
# get max and min values of the output
print("Max value: ", np.max(result))
print("Min value: ", np.min(result))