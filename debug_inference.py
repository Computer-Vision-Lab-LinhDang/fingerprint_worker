#!/usr/bin/env python3
"""
Debug script to test TensorRT inference on Jetson Nano.
Run this directly on Jetson: python3 debug_inference.py

This script tests 3 different approaches:
  1. ctypes CUDA (current)
  2. pycuda (if available)
  3. trt.cuda (using tensorrt's own cuda module if available)
"""

import ctypes
import os
import sys
import time

import numpy as np
import cv2

print("=" * 60)
print("  TensorRT Inference Debug Script")
print("=" * 60)

# ── 1. Check dependencies ──────────────────────────────────
print("\n[1] Checking dependencies...")
print("  Python:", sys.version)
print("  NumPy:", np.__version__)
print("  OpenCV:", cv2.__version__)

import tensorrt as trt
print("  TensorRT:", trt.__version__)

has_pycuda = False
try:
    import pycuda.driver as cuda
    import pycuda.autoinit
    has_pycuda = True
    print("  PyCUDA: YES")
except ImportError:
    print("  PyCUDA: NO")

# ── 2. Find model & sample image ───────────────────────────
print("\n[2] Looking for model and sample data...")

model_dir = os.path.join(os.getcwd(), "models")
sample_dir = os.path.join(os.getcwd(), "data", "sample")

# Find first .onnx file
onnx_path = None
trt_path = None
for root, dirs, files in os.walk(model_dir):
    for f in files:
        if f.endswith(".onnx"):
            onnx_path = os.path.join(root, f)
            trt_path = onnx_path.replace(".onnx", ".trt")
            break

if not onnx_path:
    print("  ERROR: No .onnx model found in", model_dir)
    sys.exit(1)

print("  ONNX model:", onnx_path)
print("  TRT engine:", trt_path, "exists:", os.path.exists(trt_path or ""))

# Find first image
sample_img = None
for f in sorted(os.listdir(sample_dir)):
    if f.endswith((".tif", ".png", ".jpg", ".bmp")):
        sample_img = os.path.join(sample_dir, f)
        break

if not sample_img:
    print("  ERROR: No images found in", sample_dir)
    sys.exit(1)

print("  Sample image:", sample_img)

# ── 3. Load/build TRT engine ───────────────────────────────
print("\n[3] Loading TensorRT engine...")

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

if trt_path and os.path.exists(trt_path):
    print("  Loading cached .trt file...")
    runtime = trt.Runtime(TRT_LOGGER)
    with open(trt_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
else:
    print("  No .trt cache, building from ONNX (may take minutes)...")
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print("  PARSE ERROR:", parser.get_error(i))
            sys.exit(1)

    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 28

    # Handle dynamic shapes
    profile = builder.create_optimization_profile()
    has_dynamic = False
    for i in range(network.num_inputs):
        inp = network.get_input(i)
        shape = inp.shape
        if any(d == -1 for d in shape):
            has_dynamic = True
            min_s = tuple(1 if d == -1 else d for d in shape)
            opt_s = tuple(1 if d == -1 else d for d in shape)
            max_s = tuple(8 if (d == -1 and idx == 0) else (1 if d == -1 else d)
                          for idx, d in enumerate(shape))
            profile.set_shape(inp.name, min_s, opt_s, max_s)
            print("  Dynamic input '{}': {} -> opt {}".format(inp.name, shape, opt_s))
    if has_dynamic:
        config.add_optimization_profile(profile)

    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("  FP16 enabled")

    engine = builder.build_engine(network, config)
    if engine is None:
        print("  ERROR: Failed to build engine")
        sys.exit(1)

    with open(trt_path, "wb") as f:
        f.write(engine.serialize())
    print("  Engine saved to", trt_path)

print("  Engine loaded!")

# ── 4. Print engine info ───────────────────────────────────
print("\n[4] Engine bindings:")
for i in range(engine.num_bindings):
    name = engine.get_binding_name(i)
    shape = engine.get_binding_shape(i)
    dtype = engine.get_binding_dtype(i)
    is_input = engine.binding_is_input(i)
    print("  [{}] '{}': shape={}, dtype={}, {}".format(
        i, name, shape, dtype, "INPUT" if is_input else "OUTPUT"))

# ── 5. Prepare input ──────────────────────────────────────
print("\n[5] Preparing input...")

context = engine.create_execution_context()

# Get input shape
input_shape = engine.get_binding_shape(0)
if any(d == -1 for d in input_shape):
    opt_shape = engine.get_profile_shape(0, 0)[1]
    context.set_binding_shape(0, opt_shape)
    input_shape = opt_shape
    print("  Set dynamic input to:", input_shape)

output_shape = context.get_binding_shape(1)
print("  Input shape:", input_shape)
print("  Output shape:", output_shape)

# Read and preprocess image
img = cv2.imread(sample_img, cv2.IMREAD_GRAYSCALE)
print("  Raw image: shape={}, dtype={}, min={}, max={}".format(
    img.shape, img.dtype, img.min(), img.max()))

_, _, target_h, target_w = input_shape
img_resized = cv2.resize(img, (target_w, target_h))
img_float = img_resized.astype(np.float32) / 255.0
input_data = img_float.reshape(input_shape).copy()
input_data = np.ascontiguousarray(input_data, dtype=np.float32)

print("  Preprocessed: shape={}, min={:.4f}, max={:.4f}, mean={:.4f}".format(
    input_data.shape, input_data.min(), input_data.max(), input_data.mean()))

# ── 6. Test with pycuda (if available) ─────────────────────
if has_pycuda:
    print("\n[6a] Testing with PyCUDA...")

    d_input = cuda.mem_alloc(input_data.nbytes)
    output_data = np.zeros(output_shape, dtype=np.float32)
    d_output = cuda.mem_alloc(output_data.nbytes)

    cuda.memcpy_htod(d_input, input_data)

    success = context.execute_v2([int(d_input), int(d_output)])
    print("  execute_v2:", success)

    cuda.memcpy_dtoh(output_data, d_output)

    result = output_data.flatten()
    print("  Output: min={:.6f}, max={:.6f}, mean={:.6f}".format(
        result.min(), result.max(), result.mean()))
    print("  First 10 values:", result[:10])
    print("  Non-zero count:", np.count_nonzero(result))

    d_input.free()
    d_output.free()
else:
    print("\n[6a] PyCUDA not available, skipping.")

# ── 7. Test with ctypes CUDA ──────────────────────────────
print("\n[6b] Testing with ctypes CUDA...")

cudart = ctypes.CDLL("libcudart.so")

cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
cudart.cudaMalloc.restype = ctypes.c_int
cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
cudart.cudaMemcpy.restype = ctypes.c_int
cudart.cudaFree.argtypes = [ctypes.c_void_p]
cudart.cudaFree.restype = ctypes.c_int
cudart.cudaDeviceSynchronize.restype = ctypes.c_int

d_input = ctypes.c_void_p()
d_output = ctypes.c_void_p()

err = cudart.cudaMalloc(ctypes.byref(d_input), input_data.nbytes)
print("  cudaMalloc input:  err={}, ptr={}".format(err, d_input.value))
err = cudart.cudaMalloc(ctypes.byref(d_output), ctypes.c_size_t(np.prod(output_shape) * 4))
print("  cudaMalloc output: err={}, ptr={}".format(err, d_output.value))

# H2D
input_ptr = input_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
err = cudart.cudaMemcpy(d_input, input_ptr, ctypes.c_size_t(input_data.nbytes), ctypes.c_int(1))
print("  cudaMemcpy H2D: err={}".format(err))

# Re-set binding shape (important for dynamic)
if any(d == -1 for d in engine.get_binding_shape(0)):
    context.set_binding_shape(0, input_data.shape)

# Execute
success = context.execute_v2([int(d_input.value), int(d_output.value)])
print("  execute_v2: {}".format(success))

cudart.cudaDeviceSynchronize()

# D2H
output_data = np.zeros(output_shape, dtype=np.float32)
output_ptr = output_data.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
err = cudart.cudaMemcpy(output_ptr, d_output, ctypes.c_size_t(output_data.nbytes), ctypes.c_int(2))
print("  cudaMemcpy D2H: err={}".format(err))

cudart.cudaFree(d_input)
cudart.cudaFree(d_output)

result = output_data.flatten()
print("  Output: min={:.6f}, max={:.6f}, mean={:.6f}".format(
    result.min(), result.max(), result.mean()))
print("  First 10 values:", result[:10])
print("  Non-zero count:", np.count_nonzero(result))

# ── 8. Summary ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("  DEBUG COMPLETE — send this output back!")
print("=" * 60)
