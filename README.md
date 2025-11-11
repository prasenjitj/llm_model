# Qwen VL API

A high-performance, enterprise-ready FastAPI application for generating responses using the Qwen2.5-VL-3B-Instruct vision-language model. This API supports text and image inputs, with built-in rate limiting, monitoring, and error handling for reliable production deployment.

## Features

- **Vision-Language Processing**: Generate responses based on text prompts and optional uploaded images.
- **Rate Limiting**: Prevents abuse with configurable per-IP and per-endpoint limits.
- **Monitoring**: Integrated Prometheus metrics for request counts, latency, and error tracking.
- **Health Checks**: Built-in `/health` endpoint for liveness probes.
- **Error Handling**: Comprehensive exception handling with structured logging.
- **Security**: Input validation, file type checks, and size limits.
- **Scalability**: Optimized for GPU usage with memory management and asynchronous processing.
- **Configuration**: Environment variable-based configuration for easy deployment.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd llm_model
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set environment variables (optional):
   ```bash
   export MODEL_NAME="Qwen/Qwen2.5-VL-3B-Instruct"
   export MAX_IMAGE_SIZE="10485760"  # 10MB in bytes
   ```

## Usage

Run the application:
```bash
python app.py
```

Or using uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

## API Endpoints

### POST /generate
Generate a response based on text and optional image input.

**Parameters:**
- `text` (required): The text prompt.
- `image` (optional): Uploaded image file (multipart/form-data).

**Rate Limit:** 10 requests per minute.

**Example Request:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -F "text=Describe this image" \
     -F "image=@example.jpg"
```

**Example with your image file:**
```bash
curl -X POST "http://localhost:8000/generate" \
     -F "text=Describe this image in detail" \
     -F "image=@/home/prasenjitjana/llm_model/image1.jpg"
```

**Response:**
```json
{
  "response": "The image shows..."
}
```

### GET /health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy"
}
```

### GET /metrics
Prometheus metrics endpoint.

## Configuration

Configure the application using environment variables:

- `MODEL_NAME`: Hugging Face model name (default: "Qwen/Qwen2.5-VL-3B-Instruct")
- `MAX_IMAGE_SIZE`: Maximum image file size in bytes (default: 10MB)

## Deployment

### Docker
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t qwen-vl-api .
docker run -p 8000:8000 qwen-vl-api
```

### Kubernetes
Use the provided Dockerfile and deploy with a Deployment and Service manifest. Configure Horizontal Pod Autoscaler based on CPU/GPU metrics.

### Production Considerations
- Use a reverse proxy (e.g., Nginx) for SSL termination and additional rate limiting.
- Deploy behind a load balancer for high availability.
- Monitor with Prometheus/Grafana and set up alerts.
- Use GPU-enabled instances for better performance.
- Implement authentication (e.g., API keys) for production use.

## System Service Setup

To run the application as a systemd service that auto-starts on system reboot, follow these steps. This ensures the API runs in the background and restarts automatically on failure.

### Prerequisites
- Linux system with systemd (most distributions).
- Sudo access.
- Virtual environment set up in the project directory.
- Valid user account (e.g., your login user) with access to the project and GPU (if applicable).

### Steps
1. **Create the Service File**:
   ```bash
   sudo nano /etc/systemd/system/qwen-vl-api.service
   ```
   Paste the following (replace paths and user as needed):
   ```
   [Unit]
   Description=Qwen VL API Service
   After=network.target

   [Service]
   Type=simple
   User=your-username  # Replace with your username (e.g., prasenjitjana)
   WorkingDirectory=/home/your-username/llm_model  # Full path to project
   ExecStart=/bin/bash -c 'source /home/your-username/llm_model/venv/bin/activate && python3 /home/your-username/llm_model/app.py'
   Restart=always
   RestartSec=5
   Environment=PATH=/home/your-username/llm_model/venv/bin
   StandardOutput=journal
   StandardError=journal

   [Install]
   WantedBy=multi-user.target
   ```
   Save and exit.

2. **Reload and Enable**:
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable qwen-vl-api.service
   sudo systemctl start qwen-vl-api.service
   ```

3. **Verify**:
   ```bash
   sudo systemctl status qwen-vl-api.service
   ```
   Test: `curl http://localhost:8000/health`.

### Managing the Service
- **Stop**: `sudo systemctl stop qwen-vl-api.service`
- **Restart**: `sudo systemctl restart qwen-vl-api.service`
- **Disable Auto-Start**: `sudo systemctl disable qwen-vl-api.service`
- **Remove**: `sudo systemctl stop qwen-vl-api.service && sudo rm /etc/systemd/system/qwen-vl-api.service && sudo systemctl daemon-reload`

## Monitoring

- **Metrics**: Access `/metrics` for Prometheus-compatible metrics (request counts, latency, errors).
- **Logs**: Structured logging to stdout/stderr. View with `sudo journalctl -u qwen-vl-api.service -f`.
- **Health Checks**: Use `/health` for liveness probes.
- **System Monitoring**: Use `htop` or `top` for CPU/RAM usage. Check GPU with `nvidia-smi` if applicable.
- **Prometheus/Grafana**: Integrate for dashboards and alerts on high latency or errors.

## Troubleshooting

- **Model Loading Issues**: Ensure sufficient GPU memory and correct model name. Check logs for CUDA errors.
- **Rate Limiting**: Check logs for 429 responses. Adjust limits in `app.py` if needed.
- **Image Processing Errors**: Verify image formats and sizes. Ensure PIL is installed.
- **Performance**: Monitor latency via `/metrics`. Scale resources or optimize model (e.g., quantization).
- **Service Startup Failures**:
  - **Error 217/USER**: Invalid user in service file. Verify user exists (`id your-username`) and update `User=` in the service file.
  - **Permission Issues**: Ensure the user owns the project directory (`sudo chown -R your-username:your-username /path/to/project`).
  - **Venv Issues**: Test manually: `sudo -u your-username bash -c 'source venv/bin/activate && python3 app.py'`.
  - **Logs**: View with `sudo journalctl -u qwen-vl-api.service -f` for detailed errors.
- **General Debugging**: Restart service and check status. Ensure dependencies are installed in venv.

## License

[Add your license here]