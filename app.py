import logging
# load .env file into the environment by default for local / dev usage
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv not available — assume environment variables are provided externally
    pass
import os
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest, Gauge
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import io
import requests
from typing import Optional
import asyncio
from itertools import count
from collections import deque
from datetime import datetime, timezone, timedelta
from colorama import Fore, Style, init

# Define IST timezone
IST = timezone(timedelta(hours=5, minutes=30))
init(autoreset=True)

# Configure logging
class ISTFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, tz=timezone.utc).astimezone(IST)
        if datefmt:
            return dt.strftime(datefmt)
        else:
            return dt.isoformat()

    def format(self, record):
        level_colors = {
            'DEBUG': Fore.CYAN,
            'INFO': Fore.GREEN,
            'WARNING': Fore.YELLOW,
            'ERROR': Fore.RED,
            'CRITICAL': Fore.MAGENTA
        }
        color = level_colors.get(record.levelname, Fore.WHITE)
        record.levelname = f"{color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

formatter = ISTFormatter('[%(asctime)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S IST')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# Load configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))  # 10MB
# Allow server-side image downloads by default to avoid API callers getting
# immediate 400 errors when they send an `image_url`. The behavior can still
# be disabled by explicitly setting `ENABLE_SERVER_IMAGE_DOWNLOAD=false`.
ENABLE_SERVER_IMAGE_DOWNLOAD = os.getenv("ENABLE_SERVER_IMAGE_DOWNLOAD", "true").lower() in ("1", "true", "yes")
ENABLE_SERVER_IMAGE_RESIZE = os.getenv("ENABLE_SERVER_IMAGE_RESIZE", "true").lower() in ("1", "true", "yes")
DEBUG_IMAGE_FORM = os.getenv("DEBUG_IMAGE_FORM", "false").lower() in ("1", "true", "yes")

# Mixed precision (FP16) toggle — only useful when a CUDA GPU is available
USE_FP16 = os.getenv("USE_FP16", "true").lower() in ("1", "true", "yes")

# Enable TF32 for A100 - significant speedup for matmuls
ENABLE_TF32 = os.getenv("ENABLE_TF32", "true").lower() in ("1", "true", "yes")

# Flash Attention 2 support (if available)
USE_FLASH_ATTENTION = os.getenv("USE_FLASH_ATTENTION", "true").lower() in ("1", "true", "yes")

# BetterTransformer optimization
USE_BETTERTRANSFORMER = os.getenv("USE_BETTERTRANSFORMER", "false").lower() in ("1", "true", "yes")

# torch.compile for additional speedup (PyTorch 2.0+)
COMPILE_MODEL = os.getenv("COMPILE_MODEL", "false").lower() in ("1", "true", "yes")

# Max new tokens for generation
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "256"))

# CUDA optimizations for A100
if torch.cuda.is_available():
    # Enable TF32 for faster matrix multiplications on A100
    if ENABLE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("TF32 enabled for A100 acceleration")
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    logger.info("cuDNN benchmark mode enabled")

app = FastAPI(title="Qwen VL API", version="1.0.0")

# Optional: prefer llmcompressor workflows for quantization/compatibility with AWQ exports
USE_LLMCOMPRESSOR = os.getenv("USE_LLMCOMPRESSOR", "false").lower() in ("1", "true", "yes")

# Try to import llmcompressor if requested. We do not call llmcompressor APIs
# here — quantization is typically done offline — but we provide diagnostics
# and a clear error message if the user requested it but it's not available.
try:
    import llmcompressor  # type: ignore
    HAS_LLMCOMPRESSOR = True
    logger.info("llmcompressor available")
except Exception:
    HAS_LLMCOMPRESSOR = False
    if USE_LLMCOMPRESSOR:
        logger.warning("USE_LLMCOMPRESSOR requested but 'llmcompressor' is not importable. Install via `pip install llmcompressor`.")

# If the model name suggests it's an AWQ-quantized artifact, disable FP16
# because mixing AMP/float16 with some AWQ/quantized formats can produce
# NaNs and CUBLAS failures. This automatic fallback reduces surprising OOMs/errs.
if 'awq' in MODEL_NAME.lower():
    if USE_FP16:
        logger.warning("MODEL_NAME contains 'awq' — disabling USE_FP16 to avoid FP16/quantized incompatibilities")
    USE_FP16 = False

# Request serial generator (simple incremental counter)
_request_serial_counter = count(1)

# Rate limiting setup - high throughput for A100
limiter = Limiter(key_func=get_remote_address, default_limits=["1000 per minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Prometheus metrics - use get_or_create pattern to avoid duplicate registration
from prometheus_client import REGISTRY

def get_or_create_counter(name, description, labelnames):
    try:
        return Counter(name, description, labelnames)
    except ValueError:
        # Already registered, get existing
        return REGISTRY._names_to_collectors.get(name + '_total') or REGISTRY._names_to_collectors.get(name)

def get_or_create_histogram(name, description, labelnames=None, buckets=None):
    try:
        kwargs = {}
        if labelnames:
            kwargs['labelnames'] = labelnames
        if buckets:
            kwargs['buckets'] = buckets
        return Histogram(name, description, **kwargs)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

def get_or_create_gauge(name, description):
    try:
        return Gauge(name, description)
    except ValueError:
        return REGISTRY._names_to_collectors.get(name)

REQUEST_COUNT = get_or_create_counter('request_count', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = get_or_create_histogram('request_latency_seconds', 'Request latency', ['method', 'endpoint'],
                            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0])
INFERENCE_QUEUE_GAUGE = get_or_create_gauge('inference_queue_size', 'Number of queued inference requests')
GPU_MEMORY_GAUGE = get_or_create_gauge('gpu_memory_used_bytes', 'GPU memory used in bytes')
BATCH_SIZE_HISTOGRAM = get_or_create_histogram('batch_size', 'Actual batch sizes processed', buckets=[1, 2, 4, 8, 16])


def update_gpu_metrics():
    """Update GPU memory usage metric."""
    if torch.cuda.is_available():
        try:
            memory_used = torch.cuda.memory_allocated()
            GPU_MEMORY_GAUGE.set(memory_used)
        except Exception:
            pass


# Concurrency settings - must be defined before inference_worker
MAX_CONCURRENT_GENERATIONS = int(os.getenv("MAX_CONCURRENT_GENERATIONS", "16"))
MAX_INFERENCE_QUEUE_SIZE = int(os.getenv("MAX_INFERENCE_QUEUE_SIZE", "500"))
# Note: inference_queue will be initialized in startup event (asyncio.Queue needs running loop)
inference_queue: asyncio.Queue = None  # type: ignore
INFERENCE_WORKERS: list = []
MAX_INFERENCE_BATCH_SIZE = int(os.getenv("MAX_INFERENCE_BATCH_SIZE", "16"))
BATCH_WAIT_TIMEOUT = float(os.getenv("BATCH_WAIT_TIMEOUT", "0.03"))
GEN_TIME_HISTORY = deque(maxlen=1000)
GEN_TIME_DEFAULT_EST = float(os.getenv("GEN_TIME_DEFAULT_EST", "3.0"))


async def inference_worker(worker_id: int):
    logger.info(f"Starting inference worker {worker_id}")
    loop = asyncio.get_running_loop()
    while True:
        # Wait for at least one request
        first_item = await inference_queue.get()
        batch = [first_item]
        # try to collect more items up to batch size, waiting a small time
        while len(batch) < MAX_INFERENCE_BATCH_SIZE:
            try:
                # attempt immediate get first to gather hot items
                item = inference_queue.get_nowait()
                batch.append(item)
            except asyncio.QueueEmpty:
                # no immediate item — wait a little for more to arrive
                try:
                    item = await asyncio.wait_for(inference_queue.get(), timeout=BATCH_WAIT_TIMEOUT)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

        # At this point 'batch' contains 1..MAX_INFERENCE_BATCH_SIZE items
        INFERENCE_QUEUE_GAUGE.set(inference_queue.qsize())
        BATCH_SIZE_HISTOGRAM.observe(len(batch))
        serials = [it[0] for it in batch]
        inputs_list = [it[1] for it in batch]
        futures = [it[2] for it in batch]
        logger.info(f"Worker {worker_id} processing serials={serials}, batch_size={len(batch)}, queue_size={inference_queue.qsize()}")

        try:
            # Prepare batched inputs by concatenating tensor values along dim=0 where compatible.
            batched_inputs = {}
            keys = set().union(*(inp.keys() for inp in inputs_list))

            # determine if we can safely batch: for every key that contains tensors,
            # Determine whether we can safely batch all inputs together.
            # Requirements to batch:
            #  - every key must be present in every input
            #  - if a key contains tensors, all values for that key must be tensors
            #    and have identical non-batch shapes and non-zero dimensions
            can_batch = True
            for k in keys:
                # collect values for this key (some inputs may omit the key)
                values = []
                missing = False
                for inp in inputs_list:
                    if k not in inp:
                        missing = True
                        break
                    values.append(inp[k])
                if missing:
                    # If any key is missing in any input, we cannot safely batch because
                    # different inputs would produce tensors/lists with mismatched batch sizes
                    logger.debug(f"Worker {worker_id} cannot batch: key '{k}' missing in some inputs")
                    can_batch = False
                    break

                # if all are tensors, check shape compatibility (ignore batch dim)
                if all(isinstance(v, torch.Tensor) for v in values):
                    # ensure tensors are per-request single-sample (batch dim == 1),
                    # have no zero-length dimensions and share identical non-batch shapes
                    base_shape = values[0].shape[1:]
                    for v in values:
                        # reject tensors with zero in any non-batch dimension or unexpected batch sizes
                        if v.ndim == 0:
                            logger.debug(f"Worker {worker_id} cannot batch: key '{k}' has scalar tensor")
                            can_batch = False
                            break
                        if v.shape[0] == 0:
                            logger.debug(f"Worker {worker_id} cannot batch: key '{k}' contains empty batch dimension shape={tuple(v.shape)}")
                            can_batch = False
                            break
                        # require per-request tensors to have batch size exactly 1 so batching is well-defined
                        if v.shape[0] != 1:
                            logger.debug(f"Worker {worker_id} cannot batch: key '{k}' expects per-request batch dim==1 but found {tuple(v.shape)}")
                            can_batch = False
                            break
                        if any(d == 0 for d in v.shape[1:]):
                            logger.debug(f"Worker {worker_id} cannot batch: key '{k}' contains zero-length non-batch dimension shape={tuple(v.shape)}")
                            can_batch = False
                            break
                        if v.shape[1:] != base_shape:
                            logger.debug(f"Worker {worker_id} cannot batch: key '{k}' non-batch shapes differ: {base_shape} vs {tuple(v.shape[1:])}")
                            can_batch = False
                            break
                if not can_batch:
                    break

            if not can_batch:
                # Incompatible shapes detected — fallback to per-item processing below
                logger.info(f"Worker {worker_id} batching disabled for serials={serials} due to incompatible inputs")
                batched_inputs = None
            else:
                # all keys present and (for tensor keys) shapes compatible — build batched tensors
                for k in keys:
                    # since we required every input to contain every key, this list will have length == len(inputs_list)
                    values = [inp[k] for inp in inputs_list]
                    # require homogeneity: either all tensors or all same non-tensor type
                    if all(isinstance(v, torch.Tensor) for v in values):
                        # concat along batch dim
                        batched_inputs[k] = torch.cat(values, dim=0)
                    else:
                        # non-tensor values must be homogeneous (same type) to batch safely
                        first_type = type(values[0])
                        if not all(isinstance(v, first_type) for v in values):
                            logger.debug(f"Worker {worker_id} cannot batch: key '{k}' contains mixed types: {[type(v) for v in values]}")
                            # this indicates inputs differ structurally — bail out of batching entirely
                            batched_inputs = None
                            break
                        batched_inputs[k] = values


            if batched_inputs is not None:
                # Move to device and match model dtype for floating tensors to avoid
                # fp16/fp32 mismatches in low-level kernels (e.g. Triton/CuBLAS).
                def _move_tensor(t: torch.Tensor):
                    if not isinstance(t, torch.Tensor):
                        return t
                    # keep integer types (input_ids, attention_mask) as their original dtype
                    if torch.is_floating_point(t):
                        try:
                            return t.to(device=model.device, dtype=model.dtype, non_blocking=True)
                        except Exception:
                            return t.to(device=model.device, non_blocking=True)
                    else:
                        return t.to(device=model.device, non_blocking=True)

                batched_inputs = {k: (_move_tensor(v) if isinstance(v, torch.Tensor) else v) for k, v in batched_inputs.items()}

                # run generation in executor to keep event loop responsive
                def blocking_generate(local_inputs):
                    with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_FP16):
                        ids = model.generate(**local_inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)
                    return ids

                gen_start = time.perf_counter()
                try:
                    generated = await loop.run_in_executor(None, blocking_generate, batched_inputs)
                    gen_elapsed = time.perf_counter() - gen_start
                    update_gpu_metrics()
                except Exception as e:
                    # If batched generation fails due to unexpected internal tensor ops
                    # (e.g., expand_as on incompatible shapes), log details and fall
                    # back to per-item generation for this batch instead of failing.
                    logger.exception(f"Worker {worker_id} batched generation failed, falling back to per-item: {e}")
                    # Force per-item code path by marking batched_inputs None
                    batched_inputs = None
                    generated = None
                    gen_elapsed = None

            # record generation duration to history for Retry-After heuristic
            # and convert batched outputs into per-request outputs only when a
            # successful batched generation actually happened.
            if batched_inputs is not None and 'generated' in locals() and generated is not None:
                try:
                    if gen_elapsed is not None:
                        try:
                            GEN_TIME_HISTORY.append(gen_elapsed)
                        except Exception:
                            pass

                    # Ensure generated is on CPU and split into per-request outputs
                    if isinstance(generated, torch.Tensor):
                        # handle shape (batch, seq)
                        # move to cpu
                        generated = generated.cpu()
                        outputs = [generated[i].unsqueeze(0) for i in range(generated.size(0))]
                    elif isinstance(generated, (list, tuple)):
                        outputs = list(generated)
                    else:
                        # unexpected: wrap single value for all
                        outputs = [generated] * len(batch)
                except Exception as e:
                    logger.exception(f"Worker {worker_id} failed to split generated outputs: {e}")
                    # fall back to per-item handling
                    outputs = None

                # fulfill futures in order for batched outputs (only if outputs produced)
                if outputs is not None:
                    for fut, out, ser in zip(futures, outputs, serials):
                        if not fut.done() and not fut.cancelled():
                            try:
                                fut.set_result(out)
                            except Exception as e:
                                logger.warning(f"Failed to set_result for serial={ser}: {e}")
                        else:
                            logger.debug(f"Future already done/cancelled for serial={ser}; skipping")

            # If we couldn't batch (batched_inputs is None) then we must process items individually
            if batched_inputs is None or ('outputs' in locals() and outputs is None):
                outputs = []
                for idx, (inp, fut, ser) in enumerate(zip(inputs_list, futures, serials)):
                    try:
                        # move to device for this single input and match model dtype for floats
                        def _move_local(t: torch.Tensor):
                            if not isinstance(t, torch.Tensor):
                                return t
                            if torch.is_floating_point(t):
                                try:
                                    return t.to(device=model.device, dtype=model.dtype, non_blocking=True)
                                except Exception:
                                    return t.to(device=model.device, non_blocking=True)
                            else:
                                return t.to(device=model.device, non_blocking=True)

                        local_inputs = {k: (_move_local(v) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}

                        def blocking_generate_single(local_inputs):
                            with torch.no_grad(), torch.cuda.amp.autocast(enabled=USE_FP16):
                                return model.generate(**local_inputs, max_new_tokens=MAX_NEW_TOKENS, use_cache=True)

                        single_start = time.perf_counter()
                        gen_single = await loop.run_in_executor(None, blocking_generate_single, local_inputs)
                        single_elapsed = time.perf_counter() - single_start
                        try:
                            GEN_TIME_HISTORY.append(single_elapsed)
                        except Exception:
                            pass

                        # ensure on CPU
                        try:
                            gen_single = gen_single.cpu()
                        except Exception:
                            pass

                        # set result
                        if not fut.done() and not fut.cancelled():
                            try:
                                fut.set_result(gen_single)
                            except Exception as e:
                                logger.warning(f"Failed to set_result for serial={ser}: {e}")
                        outputs.append(gen_single)
                    except Exception as e:
                        logger.exception(f"Worker {worker_id} single-item generation exception for serial={ser}: {e}")
                        if not fut.done() and not fut.cancelled():
                            try:
                                fut.set_exception(e)
                            except Exception as e2:
                                logger.warning(f"Failed to set_exception for future serial={ser}: {e2}")

        except Exception as e:
            logger.exception(f"Worker {worker_id} generation exception: {e}")
            # set exception on all futures
            for fut, ser in zip(futures, serials):
                if not fut.done() and not fut.cancelled():
                    try:
                        fut.set_exception(e)
                    except Exception as e2:
                        logger.warning(f"Failed to set_exception for future serial={ser}: {e2}")
        finally:
            # free device memory between batches
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            # mark every item in the batch as done
            for _ in batch:
                try:
                    inference_queue.task_done()
                except Exception:
                    pass
            INFERENCE_QUEUE_GAUGE.set(inference_queue.qsize())


@app.on_event("startup")
async def start_inference_workers():
    global inference_queue, INFERENCE_WORKERS
    # Initialize the queue in async context
    inference_queue = asyncio.Queue(maxsize=MAX_INFERENCE_QUEUE_SIZE)
    # Clear workers list in case of reload
    INFERENCE_WORKERS = []
    logger.info(f"Starting {MAX_CONCURRENT_GENERATIONS} inference worker(s)")
    for i in range(MAX_CONCURRENT_GENERATIONS):
        task = asyncio.create_task(inference_worker(i))
        INFERENCE_WORKERS.append(task)


@app.on_event("shutdown")
async def stop_inference_workers():
    logger.info("Shutting down inference workers")
    for task in INFERENCE_WORKERS:
        task.cancel()
    # allow in-flight tasks to finish
    await asyncio.gather(*INFERENCE_WORKERS, return_exceptions=True)

# Load model and processor with A100 optimizations
try:
    # If the user requested llmcompressor workflows but the package is not
    # available, fail early with a clear message so the operator can pip install it.
    if USE_LLMCOMPRESSOR and not HAS_LLMCOMPRESSOR:
        raise RuntimeError("Environment requests USE_LLMCOMPRESSOR but package 'llmcompressor' is not installed. Install with: pip install llmcompressor")

    # Choose dtype dynamically: prefer float16 when requested and CUDA is available.
    torch_dtype = torch.float16 if (USE_FP16 and torch.cuda.is_available()) else torch.float32

    # Model loading with attention optimization
    model_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": "auto",
    }
    
    # Try different attention implementations in order of preference
    attn_impl_used = "default"
    if USE_FLASH_ATTENTION:
        try:
            # First try Flash Attention 2
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("Attempting Flash Attention 2...")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
            attn_impl_used = "flash_attention_2"
            logger.info("Flash Attention 2 loaded successfully")
        except Exception as e:
            logger.warning(f"Flash Attention 2 failed: {e}")
            # Fall back to SDPA (Scaled Dot Product Attention) - PyTorch native, very fast
            try:
                model_kwargs["attn_implementation"] = "sdpa"
                logger.info("Falling back to SDPA attention...")
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
                attn_impl_used = "sdpa"
                logger.info("SDPA attention loaded successfully")
            except Exception as e2:
                logger.warning(f"SDPA failed: {e2}, using default attention")
                del model_kwargs["attn_implementation"]
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
                attn_impl_used = "eager"
    else:
        # Use SDPA by default (fast and stable)
        try:
            model_kwargs["attn_implementation"] = "sdpa"
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
            attn_impl_used = "sdpa"
        except Exception:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_NAME, **model_kwargs)
            attn_impl_used = "eager"
    
    # Optional: Apply BetterTransformer for inference optimization
    if USE_BETTERTRANSFORMER:
        try:
            model = model.to_bettertransformer()
            logger.info("BetterTransformer optimization applied")
        except Exception as e:
            logger.warning(f"BetterTransformer not available: {e}")
    
    # Compile model with torch.compile for additional speedup (PyTorch 2.0+)
    if COMPILE_MODEL and hasattr(torch, 'compile'):
        try:
            model = torch.compile(model, mode="reduce-overhead")
            logger.info("Model compiled with torch.compile()")
        except Exception as e:
            logger.warning(f"torch.compile failed: {e}")
    
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    
    # Log GPU info
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"GPU: {gpu_name}, Memory: {gpu_memory:.1f} GB")
    
    logger.info(f"Model loaded (USE_FP16={USE_FP16}, dtype={torch_dtype}, attn={attn_impl_used})")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise


# Health check endpoint with GPU info
@app.get("/health")
async def health_check():
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_memory_allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
            "gpu_memory_reserved_gb": round(torch.cuda.memory_reserved() / (1024**3), 2),
        }
    queue_size = inference_queue.qsize() if inference_queue else 0
    return {
        "status": "healthy",
        "queue_size": queue_size,
        "workers": MAX_CONCURRENT_GENERATIONS,
        "batch_size": MAX_INFERENCE_BATCH_SIZE,
        **gpu_info
    }

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    update_gpu_metrics()
    return generate_latest()

@app.post("/generate")
async def generate_response(
    request: Request,
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="started").inc()
    start_time = time.time()
    
    try:
        # Generate a serial number for this request and log minimal info
        req_serial = next(_request_serial_counter)
        logger.info(f"Processing request: serial={req_serial}, image_url={image_url}")
        
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
        if image and image_url:
            raise HTTPException(status_code=400, detail="Provide either image file or URL, not both")
        if image and image.filename:
            if not image.content_type.startswith("image/"):
                raise HTTPException(status_code=400, detail="Invalid image file type")
            image_data = await image.read()
            if len(image_data) > MAX_IMAGE_SIZE:
                raise HTTPException(status_code=413, detail="Image too large")
            await image.seek(0)  # Reset file pointer
        
        # Prepare inputs
        messages = [{"role": "user", "content": []}]

        img = None
        if image:
            logger.info("Loading image from upload")
            # Read image
            image_data = await image.read()
            img = Image.open(io.BytesIO(image_data))
            logger.info(f"Image loaded from upload: {img.size}")

        elif image_url:
            if not ENABLE_SERVER_IMAGE_DOWNLOAD:
                # Client-side download is enabled/expected; do not fetch here
                # Provide a clearer error that indicates how to enable server-side
                # download behavior (set the `ENABLE_SERVER_IMAGE_DOWNLOAD` env var).
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "Server is configured to reject image URLs. "
                        "To allow the server to fetch image URLs, set the environment "
                        "variable `ENABLE_SERVER_IMAGE_DOWNLOAD=true` or upload image data directly."
                    ),
                )

            logger.info(f"Fetching image from URL: {image_url}")
            try:
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                if not response.headers.get('content-type', '').startswith('image/'):
                    raise HTTPException(status_code=400, detail="Invalid image URL")
                image_data = response.content
                if len(image_data) > MAX_IMAGE_SIZE:
                    raise HTTPException(status_code=413, detail="Image too large")
                img = Image.open(io.BytesIO(image_data))
                logger.info(f"Image loaded from URL: {img.size}")
            except requests.RequestException as e:
                raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")

        if img:
            # Optionally resize image server-side if resolution exceeds 800x600
            width, height = img.size
            if ENABLE_SERVER_IMAGE_RESIZE and width * height > 800 * 600:
                logger.info(f"Resizing image from {img.size} to 512x512")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                logger.info(f"Image resized to: {img.size}")
            else:
                if not ENABLE_SERVER_IMAGE_RESIZE:
                    logger.info("Server-side image resize disabled; using client-provided image size")

            messages[0]["content"].append({"type": "image", "image": img})
        else:
            # image is None and image_url is None; optionally log form details for debugging
            if DEBUG_IMAGE_FORM:
                content_type = request.headers.get("content-type", "(none)")
                logger.info(f"Debug: serial={req_serial} content-type={content_type}")
                try:
                    # only attempt to read form data for form-encoded or multipart requests
                    if "multipart/form-data" in content_type or "application/x-www-form-urlencoded" in content_type:
                        form = await request.form()
                        # show the keys and the raw image_url form field if present
                        logger.info(f"Debug: serial={req_serial} form-keys={list(form.keys())}")
                        logger.info(f"Debug: serial={req_serial} raw-image_url={form.get('image_url')}")
                    else:
                        logger.info(f"Debug: serial={req_serial} request likely not form-encoded, use -F or set Content-Type to multipart/form-data")
                except Exception as e:
                    logger.warning(f"Debug: serial={req_serial} form read failed: {e}")

        messages[0]["content"].append({"type": "text", "text": text})

        # Process inputs
        logger.info("Processing inputs with processor")
        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            # build tensors in CPU; workers will move them to `model.device`
            inputs = processor(text=text_input, images=img if img else None,
                               return_tensors="pt")
        except Exception as e:
            logger.error(f"Input processing failed: {e}")
            raise HTTPException(status_code=500, detail="Input processing error")

        # Generate response — enqueue for worker pool. If the queue is full, return 429.
        logger.info("Enqueuing request for generation")
        if inference_queue is None:
            raise HTTPException(status_code=503, detail="Server not ready - queue not initialized")
        req_future = asyncio.get_running_loop().create_future()
        try:
            inference_queue.put_nowait((req_serial, inputs, req_future))
            INFERENCE_QUEUE_GAUGE.set(inference_queue.qsize())
            logger.info(f"Enqueued request: serial={req_serial}, queue_size={inference_queue.qsize()}")
        except asyncio.QueueFull:
            # Estimate retry-after using average generation time and queue depth
            queue_len = inference_queue.qsize()
            try:
                avg = sum(GEN_TIME_HISTORY) / len(GEN_TIME_HISTORY) if len(GEN_TIME_HISTORY) else GEN_TIME_DEFAULT_EST
            except Exception:
                avg = GEN_TIME_DEFAULT_EST
            # estimate how many batches ahead we are: use queue_len / workers
            est_seconds = (queue_len / max(1, MAX_CONCURRENT_GENERATIONS)) * max(0.1, avg)
            retry_after = str(int(est_seconds) + 1)
            logger.warning(f"Queue full, rejecting request: serial={req_serial} retry_after={retry_after}s")
            raise HTTPException(status_code=429, detail="Server overloaded — try again later", headers={"Retry-After": retry_after})

        try:
            generated_ids = await req_future
        except Exception as e:
            logger.error(f"Generation worker failed: {e}")
            raise HTTPException(status_code=500, detail="Generation error")

        # Decode response
        logger.info("Decoding response")
        response = processor.batch_decode(
            generated_ids, skip_special_tokens=True)[0]
        # Extract only the assistant's response
        response = response.split("assistant\n")[-1].strip()
        logger.info(f"Final response: {response}")
        # Clear cache after decoding to free VRAM
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="success").inc()
        return {"response": response}

    except HTTPException:
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="error").inc()
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="error").inc()
        return JSONResponse(status_code=500, content={"error": "Internal server error"})
    finally:
        elapsed_time = time.time() - start_time
        logger.info(f"Request processed in {elapsed_time:.2f} seconds")
        REQUEST_LATENCY.labels(method="POST", endpoint="/generate").observe(elapsed_time)

if __name__ == "__main__":
    import uvicorn
    # Use the app object directly to avoid re-importing the module
    port = int(os.getenv("PORT", "8000"))
    logger.info(f"Starting server on port {port}")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
    )
