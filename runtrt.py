import tensorrt as trt
import numpy as np
import os

logger = trt.Logger(trt.Logger.WARNING)

builder = trt.Builder(logger)

network = builder.create_network()

with open('trtfp32.rgb_320.retrained.engine', 'rb') as f:
    engine_data = f.read()
    engine = builder.deserialize_cuda_engine(engine_data)

context = engine.create_execution_context()