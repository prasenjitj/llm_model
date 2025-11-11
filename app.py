import logging
import os
import time
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from PIL import Image
import io
import requests
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load configuration from environment variables
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-VL-3B-Instruct")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", 10 * 1024 * 1024))  # 10MB
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", 30))  # seconds

app = FastAPI(title="Qwen VL API", version="1.0.0")

# Rate limiting setup
limiter = Limiter(key_func=get_remote_address, default_limits=["100 per minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'Request latency', ['method', 'endpoint'])

# Load model and processor with error handling
try:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="auto", load_in_8bit=True
    )
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    logger.info("Model and processor loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return generate_latest()

@app.post("/generate")
@limiter.limit("10 per second")  # Stricter limit for this endpoint
async def generate_response(
    request: Request,
    text: str = Form(...),
    image: Optional[UploadFile] = File(None),
    image_url: Optional[str] = Form(None)
):
    REQUEST_COUNT.labels(method="POST", endpoint="/generate", status="started").inc()
    start_time = time.time()
    
    try:
        logger.info(f"Processing request: text={text[:50]}..., image={image is not None}, url={image_url}")
        
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text input is required")
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
            logger.info("Downloading image from URL")
            # Download image from URL
            try:
                response = requests.get(image_url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                img = Image.open(io.BytesIO(response.content))
                logger.info(f"Image downloaded from URL: {img.size}")
            except requests.RequestException as e:
                logger.error(f"Failed to download image: {e}")
                raise HTTPException(status_code=400, detail="Invalid image URL")

        if img:
            # Resize image if resolution exceeds 640x480
            width, height = img.size
            if width * height > 640 * 480:
                logger.info(f"Resizing image from {img.size} to 384x384")
                img = img.resize((384, 384), Image.Resampling.LANCZOS)
                logger.info(f"Image resized to: {img.size}")
            messages[0]["content"].append({"type": "image", "image": img})

        messages[0]["content"].append({"type": "text", "text": text})

        # Process inputs
        logger.info("Processing inputs with processor")
        try:
            text_input = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            inputs = processor(text=text_input, images=img if img else None,
                               return_tensors="pt").to(model.device)
        except Exception as e:
            logger.error(f"Input processing failed: {e}")
            raise HTTPException(status_code=500, detail="Input processing error")

        # Generate response
        logger.info("Generating response")
        try:
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=512)
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            logger.error(f"Generation failed: {e}")
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
        REQUEST_LATENCY.labels(method="POST", endpoint="/generate").observe(time.time() - start_time)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, workers=2)
