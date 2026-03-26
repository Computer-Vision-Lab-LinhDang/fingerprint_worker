"""
Inference service — run fingerprint embedding models.

Based on teammate's FingerEngine pattern (jetson_nano/inference.py).

Runtime priority:
  1. ONNX Runtime (works everywhere, most compatible)
  2. TensorRT + PyCUDA (Jetson Nano GPU, optional)
  3. Mock engine (development/testing)

Dependencies:
  - numpy, cv2 (pre-installed on JetPack)
  - onnxruntime (pip install onnxruntime)
  - tensorrt + pycuda (optional, JetPack pre-installed)
"""

import json
import logging
import os
import time
import glob

import numpy as np
import cv2

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────
MODEL_DIR = os.path.join(os.getcwd(), "models")

# Model input configuration (DeepPrint)
INPUT_SIZE = (299, 299)  # (W, H)
INPUT_CHANNELS = 1       # grayscale


# ── Image Preprocessing ─────────────────────────────────────
def preprocess_from_file(image_path, input_size=INPUT_SIZE):
    """Load image from file path and preprocess for model input."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image: {}".format(image_path))
    return _preprocess(img, input_size)


def preprocess_from_bytes(image_bytes, input_size=INPUT_SIZE):
    """Decode image bytes and preprocess for model input."""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Failed to decode image bytes")

    # Convert to grayscale if needed
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)

    return _preprocess(img, input_size)


def _preprocess(img_gray, input_size):
    """
    Core preprocessing: resize, normalize, reshape.
    Input: (H, W) uint8 grayscale
    Output: (1, 1, H, W) float32 normalized to [0, 1]
    """
    img = cv2.resize(img_gray, input_size, interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    h, w = img.shape
    img = img.reshape(1, 1, h, w)  # (1, 1, H, W)
    return img


# ── ONNX Runtime Inference (Primary) ────────────────────────
class ONNXInference(object):
    """Primary runtime — ONNX Runtime with GPU/CPU providers."""

    def __init__(self, onnx_path):
        self.onnx_path = onnx_path
        self.session = None
        self.input_name = None
        self.input_shape = None
        self.output_shape = None
        self.backend = "onnx"

    def load(self, progress_callback=None):
        """Load ONNX model with best available provider."""
        import onnxruntime as ort

        available = ort.get_available_providers()
        if progress_callback:
            progress_callback("Available providers: {}".format(available))

        providers = []
        if 'TensorrtExecutionProvider' in available:
            providers.append('TensorrtExecutionProvider')
        if 'CUDAExecutionProvider' in available:
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')

        if progress_callback:
            progress_callback("Loading ONNX model with {}...".format(providers[0]))

        self.session = ort.InferenceSession(self.onnx_path, providers=providers)

        inp = self.session.get_inputs()[0]
        out = self.session.get_outputs()[0]
        self.input_name = inp.name
        self.input_shape = inp.shape
        self.output_shape = out.shape

        if progress_callback:
            progress_callback("Model loaded | input: {} | output: {}".format(
                self.input_shape, self.output_shape))

    def infer(self, preprocessed):
        """Run inference. Returns raw output array."""
        outputs = self.session.run(None, {self.input_name: preprocessed})
        return outputs[0].astype(np.float32)


# ── TensorRT Inference (Optional, needs pycuda) ─────────────
class TensorRTInference(object):
    """
    Optional GPU runtime — TensorRT + PyCUDA.
    Requires pre-converted .trt engine file.
    Based on teammate's working pattern with page-locked memory.
    """

    def __init__(self, trt_path):
        self.trt_path = trt_path
        self.engine = None
        self.context = None
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = None
        self.backend = "trt"

    def load(self, progress_callback=None):
        """Load TensorRT engine with PyCUDA buffers."""
        import tensorrt as trt
        import pycuda.driver as cuda
        import pycuda.autoinit  # noqa: F401

        if progress_callback:
            progress_callback("Loading TensorRT engine...")

        trt_logger = trt.Logger(trt.Logger.WARNING)
        with open(self.trt_path, "rb") as f:
            runtime = trt.Runtime(trt_logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        # Allocate page-locked host memory + device memory
        self._inputs = []
        self._outputs = []
        self._bindings = []
        self._stream = cuda.Stream()

        for binding in self.engine:
            shape = self.engine.get_binding_shape(binding)
            size = trt.volume(shape)
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self._bindings.append(int(device_mem))
            if self.engine.binding_is_input(binding):
                self._inputs.append({"host": host_mem, "device": device_mem})
            else:
                self._outputs.append({"host": host_mem, "device": device_mem})

        if progress_callback:
            progress_callback("TensorRT engine loaded with PyCUDA buffers")

    def infer(self, preprocessed):
        """Run inference with TensorRT. Returns raw output array."""
        import pycuda.driver as cuda

        # Copy input to page-locked host buffer
        np.copyto(self._inputs[0]["host"], preprocessed.ravel())

        # Transfer to GPU
        for inp in self._inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self._stream)

        # Execute
        self.context.execute_async_v2(
            bindings=self._bindings,
            stream_handle=self._stream.handle,
        )

        # Transfer outputs back
        for out in self._outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self._stream)

        self._stream.synchronize()

        return self._outputs[0]["host"].astype(np.float32)


# ── Mock Inference (Development) ────────────────────────────
class MockInference(object):
    """Mock engine for development/testing without a real model."""

    def __init__(self, embedding_dim=192):
        self.embedding_dim = embedding_dim
        self.backend = "mock"

    def load(self, progress_callback=None):
        if progress_callback:
            progress_callback("Using MOCK engine (random embeddings for testing)")

    def infer(self, preprocessed):
        np.random.seed(int(np.sum(preprocessed * 1000) % (2**31)))
        return np.random.randn(self.embedding_dim).astype(np.float32)


# ── Embedding Utilities ─────────────────────────────────────
def normalize_embedding(embedding):
    """L2-normalize embedding to unit length (required for cosine similarity)."""
    flat = embedding.flatten()
    norm = np.linalg.norm(flat)
    if norm > 1e-10:
        flat = flat / norm
    return flat


def compress_embedding(embedding):
    """
    Compress float32 embedding to 8-bit integers + scaling params.
    Returns: (bytes, min, max)
    """
    emb_min = float(embedding.min())
    emb_max = float(embedding.max())
    emb_range = emb_max - emb_min
    if emb_range < 1e-10:
        emb_range = 1.0
    quantized = ((embedding - emb_min) / emb_range * 255.0).astype(np.uint8)
    return quantized.tobytes(), emb_min, emb_max


def decompress_embedding(compressed_bytes, emb_min, emb_max):
    """Decompress 8-bit bytes back to float32 embedding, re-normalized."""
    quantized = np.frombuffer(compressed_bytes, dtype=np.uint8).astype(np.float32)
    emb_range = emb_max - emb_min
    if emb_range < 1e-10:
        emb_range = 1.0
    embedding = quantized / 255.0 * emb_range + emb_min
    return normalize_embedding(embedding)


# ── Convert ONNX → TensorRT using trtexec ──────────────────
TRTEXEC_PATH = "/usr/src/tensorrt/bin/trtexec"


def convert_onnx_to_trt(onnx_path, trt_path, fp16=True, progress_callback=None):
    """
    Convert ONNX model to TensorRT engine using trtexec (NVIDIA official tool).
    Same approach as teammate's convert.sh script.
    """
    import subprocess

    if os.path.exists(trt_path):
        if progress_callback:
            progress_callback("TensorRT engine found (cached): {}".format(
                os.path.basename(trt_path)))
        return True

    if not os.path.exists(TRTEXEC_PATH):
        logger.warning("trtexec not found at %s", TRTEXEC_PATH)
        return False

    if progress_callback:
        progress_callback("Converting ONNX -> TensorRT with trtexec (may take minutes)...")

    cmd = [
        TRTEXEC_PATH,
        "--onnx={}".format(onnx_path),
        "--saveEngine={}".format(trt_path),
    ]
    if fp16:
        cmd.append("--fp16")

    logger.info("Running: %s", " ".join(cmd))

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,  # 10 min max
        )
        if result.returncode == 0:
            if progress_callback:
                progress_callback("TensorRT engine saved: {}".format(
                    os.path.basename(trt_path)))
            return True
        else:
            output = result.stdout.decode("utf-8", errors="replace")
            logger.error("trtexec failed (code %d):\n%s", result.returncode, output[-500:])
            if progress_callback:
                progress_callback("trtexec conversion failed — falling back to ONNX Runtime")
            return False
    except Exception as exc:
        logger.error("trtexec error: %s", exc)
        return False


# ── Factory — auto-select best runtime ──────────────────────
def create_inference_engine(onnx_path):
    """
    Create the best available inference engine.
    Priority: TensorRT (trtexec) > ONNX Runtime > Mock.
    """
    trt_path = onnx_path.replace(".onnx", ".trt")

    # 1. Try TensorRT + PyCUDA (fastest)
    try:
        import tensorrt  # noqa: F401
        import pycuda     # noqa: F401

        # Auto-convert ONNX → .trt if needed (using trtexec like teammate)
        if convert_onnx_to_trt(onnx_path, trt_path):
            logger.info("TensorRT + PyCUDA — using GPU acceleration")
            return TensorRTInference(trt_path)
    except ImportError:
        logger.warning("TensorRT or PyCUDA not available")

    # 2. ONNX Runtime (fallback, more compatible)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        if 'CUDAExecutionProvider' in providers:
            logger.info("ONNX Runtime available with CUDA GPU acceleration")
        else:
            logger.info("ONNX Runtime available (CPU mode)")
        return ONNXInference(onnx_path)
    except ImportError:
        logger.warning("ONNX Runtime not available")

    # 3. Fallback: Mock
    logger.warning("No inference runtime found — using MOCK engine")
    return MockInference()


# ── Run test on sample data ────────────────────────────────
def run_sample_test(model_type, model_name, sample_dir, output_dir, progress_callback=None):
    """
    Run inference on all images in sample_dir.
    Saves results (embedding vectors) as JSON to output_dir.

    Returns: list of dicts [{filename, vector, inference_time_ms}, ...]
    """
    # Find model
    onnx_path = os.path.join(MODEL_DIR, model_type, model_name)
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
    engine.load(progress_callback)

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
            input_data = preprocess_from_file(img_path)

            # Inference
            t0 = time.time()
            raw_output = engine.infer(input_data)
            elapsed_ms = (time.time() - t0) * 1000

            # L2 normalize
            embedding = normalize_embedding(raw_output)

            result = {
                "filename": filename,
                "vector": embedding.tolist(),
                "vector_dim": len(embedding),
                "inference_time_ms": round(elapsed_ms, 2),
            }
            results.append(result)

            if progress_callback:
                progress_callback(
                    "[{}/{}] {} -> {}D vector ({:.1f}ms)".format(
                        idx + 1, len(images), filename,
                        len(embedding), elapsed_ms,
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
            "runtime": engine.backend,
            "total_images": len(images),
            "successful": sum(1 for r in results if "vector" in r),
            "failed": sum(1 for r in results if "error" in r),
            "results": results,
        }, f, indent=2)

    if progress_callback:
        progress_callback("Results saved to: {}".format(output_file))

    return results
