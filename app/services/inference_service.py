"""
Inference service — ONNX to TensorRT conversion and embedding inference.

This service handles:
1. Converting ONNX models to TensorRT engines for Jetson Nano
2. Running inference on fingerprint images
3. Returning embedding vectors

Dependencies (pre-installed on JetPack):
  - tensorrt
  - pycuda
  - numpy
  - cv2 (OpenCV)
"""

import json
import logging
import os
import time
import glob

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# Model directory
MODEL_DIR = os.path.join(os.getcwd(), "models")


# ── Image Preprocessing ─────────────────────────────────────
def preprocess_image(image_path, input_shape):
    """
    Load and preprocess a fingerprint image for inference.

    Args:
        image_path: path to .tif image
        input_shape: model input shape, e.g. [1, 1, 96, 96]

    Returns:
        numpy array ready for inference
    """
    # Read as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image: {}".format(image_path))

    # Get target H, W from model input shape
    if len(input_shape) == 4:
        _, channels, target_h, target_w = input_shape
    elif len(input_shape) == 3:
        channels, target_h, target_w = input_shape
    else:
        raise ValueError("Unexpected input shape: {}".format(input_shape))

    # Handle dynamic dimensions (None or -1 or string)
    if not isinstance(target_h, int) or target_h <= 0:
        target_h = img.shape[0]
    if not isinstance(target_w, int) or target_w <= 0:
        target_w = img.shape[1]

    # Resize
    img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    # Convert to float32 and normalize to [0, 1]
    arr = img.astype(np.float32) / 255.0

    # Reshape to (1, C, H, W)
    if len(input_shape) == 4:
        arr = arr.reshape(1, 1, target_h, target_w)
    else:
        arr = arr.reshape(1, target_h, target_w)

    return arr


# ── TensorRT Engine ─────────────────────────────────────────
class TensorRTInference(object):
    """Run inference using TensorRT engine."""

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.trt_path = onnx_path.replace(".onnx", ".trt")
        self.engine = None
        self.context = None
        self.input_shape = None
        self.output_shape = None
        self._stream = None

    def build_engine(self, progress_callback=None):
        """Convert ONNX to TensorRT engine."""
        try:
            import tensorrt as trt
        except ImportError:
            raise ImportError(
                "TensorRT is not installed. "
                "On Jetson Nano, it comes pre-installed with JetPack SDK."
            )

        if os.path.exists(self.trt_path):
            logger.info("TensorRT engine already exists: %s", self.trt_path)
            if progress_callback:
                progress_callback("TensorRT engine found (cached)")
            return self.trt_path

        if progress_callback:
            progress_callback("Converting ONNX -> TensorRT (this may take a few minutes)...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

        builder = trt.Builder(TRT_LOGGER)
        network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
        network = builder.create_network(network_flags)
        parser = trt.OnnxParser(network, TRT_LOGGER)

        # Parse ONNX
        with open(self.onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for i in range(parser.num_errors):
                    logger.error("ONNX parse error: %s", parser.get_error(i))
                raise RuntimeError("Failed to parse ONNX model")

        if progress_callback:
            progress_callback("ONNX parsed, building TensorRT engine...")

        # Build config
        config = builder.create_builder_config()
        config.max_workspace_size = 1 << 28  # 256 MB

        # Use FP16 if available (Jetson Nano supports FP16)
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            if progress_callback:
                progress_callback("FP16 mode enabled (Jetson GPU acceleration)")

        # Build engine
        engine = builder.build_engine(network, config)
        if engine is None:
            raise RuntimeError("Failed to build TensorRT engine")

        # Save engine
        with open(self.trt_path, "wb") as f:
            f.write(engine.serialize())

        logger.info("TensorRT engine saved: %s", self.trt_path)
        if progress_callback:
            progress_callback("TensorRT engine built and saved")

        return self.trt_path

    def load_engine(self, progress_callback=None):
        """Load TensorRT engine for inference."""
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "Missing dependency: {}. "
                "TensorRT and PyCUDA are pre-installed with JetPack.".format(e)
            )

        if not os.path.exists(self.trt_path):
            self.build_engine(progress_callback)

        if progress_callback:
            progress_callback("Loading TensorRT engine...")

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(TRT_LOGGER)

        with open(self.trt_path, "rb") as f:
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()
        self._stream = cuda.Stream()

        # Get input/output shapes
        for i in range(self.engine.num_bindings):
            shape = self.engine.get_binding_shape(i)
            if self.engine.binding_is_input(i):
                self.input_shape = tuple(shape)
            else:
                self.output_shape = tuple(shape)

        if progress_callback:
            progress_callback("Engine loaded | input: {} | output: {}".format(
                self.input_shape, self.output_shape,
            ))

    def infer(self, input_data):
        """Run inference on preprocessed input. Returns output numpy array."""
        import pycuda.driver as cuda

        # Allocate device memory
        d_input = cuda.mem_alloc(input_data.nbytes)
        output_data = np.empty(self.output_shape, dtype=np.float32)
        d_output = cuda.mem_alloc(output_data.nbytes)

        # Transfer input to GPU
        cuda.memcpy_htod_async(d_input, input_data, self._stream)

        # Run inference
        self.context.execute_async_v2(
            bindings=[int(d_input), int(d_output)],
            stream_handle=self._stream.handle,
        )

        # Transfer output from GPU
        cuda.memcpy_dtoh_async(output_data, d_output, self._stream)
        self._stream.synchronize()

        # Free device memory
        d_input.free()
        d_output.free()

        return output_data.flatten()


# ── ONNX Runtime Fallback (for x86 / no TensorRT) ──────────
class ONNXInference(object):
    """Fallback: run inference using ONNX Runtime (CPU/GPU)."""

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_shape = None

    def load_engine(self, progress_callback=None):
        """Load ONNX model."""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError(
                "Neither TensorRT nor ONNX Runtime is installed. "
                "Install onnxruntime: pip install onnxruntime"
            )

        if progress_callback:
            progress_callback("Loading ONNX model with ONNX Runtime...")

        self.session = ort.InferenceSession(self.onnx_path)

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.output_shape = out.shape

        # Replace dynamic dims with defaults
        resolved = []
        for d in self.input_shape:
            if isinstance(d, int) and d > 0:
                resolved.append(d)
            else:
                resolved.append(1)  # batch dim default
        self.input_shape = tuple(resolved)

        if progress_callback:
            progress_callback("Model loaded | input: {} | output: {}".format(
                self.input_shape, self.output_shape,
            ))

    def build_engine(self, progress_callback=None):
        """No-op for ONNX Runtime."""
        if progress_callback:
            progress_callback("Using ONNX Runtime (no TensorRT conversion needed)")

    def infer(self, input_data):
        """Run inference."""
        result = self.session.run(None, {self.input_name: input_data})
        return result[0].flatten()


# ── Factory — auto-select best runtime ──────────────────────
def create_inference_engine(onnx_path):
    """
    Create the best available inference engine.
    Prefers TensorRT (Jetson Nano), falls back to ONNX Runtime.
    """
    try:
        import tensorrt  # noqa: F401
        logger.info("TensorRT available — using GPU acceleration")
        return TensorRTInference(onnx_path)
    except ImportError:
        pass

    try:
        import onnxruntime  # noqa: F401
        logger.info("Using ONNX Runtime fallback")
        return ONNXInference(onnx_path)
    except ImportError:
        pass

    raise ImportError(
        "No inference runtime found. Install TensorRT (Jetson) or onnxruntime."
    )


# ── Run test on sample data ────────────────────────────────
def run_sample_test(model_type, model_name, sample_dir, output_dir, progress_callback=None):
    """
    Run inference on all .tif images in sample_dir.
    Saves results (embedding vectors) as JSON to output_dir.

    Returns: list of dicts [{filename, vector, inference_time_ms}, ...]
    """
    # Find model
    onnx_path = os.path.join(MODEL_DIR, model_type, model_name, "model.onnx")
    if not os.path.exists(onnx_path):
        raise FileNotFoundError("Model not found: {}".format(onnx_path))

    # Find images
    images = sorted(
        glob.glob(os.path.join(sample_dir, "*.tif"))
        + glob.glob(os.path.join(sample_dir, "*.png"))
        + glob.glob(os.path.join(sample_dir, "*.jpg"))
        + glob.glob(os.path.join(sample_dir, "*.bmp"))
    )
    if not images:
        raise FileNotFoundError("No images found in: {}".format(sample_dir))

    # Create output dir
    os.makedirs(output_dir, exist_ok=True)

    # Create engine
    if progress_callback:
        progress_callback("Initializing inference engine...")

    engine = create_inference_engine(onnx_path)
    engine.build_engine(progress_callback)
    engine.load_engine(progress_callback)

    if progress_callback:
        progress_callback("Starting inference on {} images...".format(len(images)))

    results = []

    for idx, img_path in enumerate(images):
        filename = os.path.basename(img_path)

        if progress_callback:
            progress_callback(
                "[{}/{}] Processing: {}".format(idx + 1, len(images), filename)
            )

        try:
            # Preprocess
            input_data = preprocess_image(img_path, engine.input_shape)

            # Inference
            t0 = time.time()
            vector = engine.infer(input_data)
            elapsed_ms = (time.time() - t0) * 1000

            result = {
                "filename": filename,
                "vector": vector.tolist(),
                "vector_dim": len(vector),
                "inference_time_ms": round(elapsed_ms, 2),
            }
            results.append(result)

            if progress_callback:
                progress_callback(
                    "[{}/{}] {} -> {}D vector ({:.1f}ms)".format(
                        idx + 1, len(images), filename,
                        len(vector), elapsed_ms,
                    )
                )

        except Exception as exc:
            logger.error("Failed to process %s: %s", filename, exc)
            results.append({
                "filename": filename,
                "error": str(exc),
            })
            if progress_callback:
                progress_callback(
                    "[{}/{}] {} -> ERROR: {}".format(
                        idx + 1, len(images), filename, exc,
                    )
                )

    # Save results
    output_file = os.path.join(output_dir, "results.json")
    with open(output_file, "w") as f:
        json.dump({
            "model_type": model_type,
            "model_name": model_name,
            "onnx_path": onnx_path,
            "runtime": engine.__class__.__name__,
            "total_images": len(images),
            "successful": sum(1 for r in results if "vector" in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results,
        }, f, indent=2)

    if progress_callback:
        progress_callback("Results saved to: {}".format(output_file))

    return results
